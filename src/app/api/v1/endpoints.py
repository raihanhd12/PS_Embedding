"""
API endpoints for the Embedding API with multi-file support
"""

import asyncio
import json
import os
import uuid
from typing import List, Optional

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)

from src.app.schemas.embedding_schema import (
    ChunkSearchResult,
    DocumentListResponse,
    DocumentResponse,
    MultiDocumentProcessResponse,
    MultiDocumentUploadResponse,
    MultiEmbeddingDocumentRequest,
    SearchRequest,
    SearchResponse,
)
from src.app.services.database_service import DatabaseService
from src.app.services.embedding_service import EmbeddingService
from src.app.services.storage_service import StorageService
from app.services.vector_database_service import VectorDatabaseService
from app.utils.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from app.utils.security import validate_api_key
from app.utils.session import get_session_id

# Create API router
router = APIRouter(prefix="/api/v1", dependencies=[Depends(validate_api_key)])

# Initialize services
embedding_service = EmbeddingService()
vector_db_service = VectorDatabaseService()
storage_service = StorageService()
db_service = DatabaseService()

@router.post("/upload/batch", response_model=MultiDocumentUploadResponse)
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(None),
    session_id: str = Depends(get_session_id),
):
    """Upload multiple documents in a single request"""
    successful = []
    failed = []

    # Parse metadata if provided
    try:
        metadata_dict = json.loads(metadata) if metadata and metadata.strip() else {}
    except json.JSONDecodeError:
        # Handle invalid JSON gracefully
        print(f"Invalid JSON metadata received: {metadata}")
        metadata_dict = {}

    metadata_dict["active"] = True  # Set default active status
    metadata_dict["session_id"] = session_id  # Add session ID

    # Check MinIO connection once
    try:
        storage_service.client.list_buckets()
        print("MinIO connection verified")
    except Exception as minio_conn_error:
        print(f"ERROR: Cannot connect to MinIO: {str(minio_conn_error)}")
        raise HTTPException(
            status_code=500,
            detail=f"Storage service unavailable: {str(minio_conn_error)}",
        )

    # Process each file
    for file in files:
        try:
            print(f"Processing upload for file: {file.filename}")

            # Read file content
            content = await file.read()

            # Upload to MinIO
            success, object_name = await storage_service.upload_file(
                content, file.filename, file.content_type, metadata_dict
            )

            if not success:
                failed.append(
                    {"filename": file.filename, "error": "Upload to storage failed"}
                )
                continue

            storage_path = object_name
            print(f"File successfully uploaded to MinIO with path: {storage_path}")

            # Save to database
            document = db_service.create_document(
                filename=file.filename,
                content_type=file.content_type,
                storage_path=storage_path,
                metadata=metadata_dict,
            )

            successful.append(
                {
                    "file_id": document.id,
                    "filename": document.filename,
                    "storage_path": document.storage_path,
                    "content_type": document.content_type,
                }
            )

        except Exception as e:
            print(f"Error processing file {file.filename}: {str(e)}")
            failed.append({"filename": file.filename, "error": str(e)})

    return MultiDocumentUploadResponse(
        successful=successful, failed=failed, total_uploaded=len(successful)
    )


@router.post("/embedding/local", response_model=MultiDocumentProcessResponse)
async def local_file_embedding(
    file_paths: List[str] = Body(..., description="List of local file paths"),
    chunk_size: Optional[int] = Body(
        DEFAULT_CHUNK_SIZE, description="Chunk size for text splitting"
    ),
    chunk_overlap: Optional[int] = Body(
        DEFAULT_CHUNK_OVERLAP, description="Chunk overlap for text splitting"
    ),
    additional_metadata: Optional[dict] = Body(
        None, description="Additional metadata for all files"
    ),
    session_id: str = Depends(get_session_id),
    _: str = Depends(validate_api_key),
):
    # Add session ID to metadata
    if additional_metadata is None:
        additional_metadata = {}
    additional_metadata["session_id"] = session_id

    successful = []
    failed = []
    total_chunks = 0

    async def process_single_local_file(file_path):
        """Process a single local file for embedding"""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {
                    "file_path": file_path,
                    "status": "error",
                    "message": "File not found on server",
                }

            # Get file metadata
            filename = os.path.basename(file_path)
            _, file_extension = os.path.splitext(filename)

            # Determine content type based on extension
            content_type_map = {
                ".pdf": "application/pdf",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".doc": "application/msword",
                ".txt": "text/plain",
                ".md": "text/markdown",
                ".html": "text/html",
                ".htm": "text/html",
                ".csv": "text/csv",
                ".json": "application/json",
            }

            content_type = content_type_map.get(
                file_extension.lower(), "application/octet-stream"
            )

            # Read file content
            try:
                with open(file_path, "rb") as f:
                    file_content = f.read()
            except Exception as file_error:
                return {
                    "file_path": file_path,
                    "status": "error",
                    "message": f"Error reading file: {str(file_error)}",
                }

            # Generate a document ID (without storing in database)
            document_id = str(uuid.uuid4())

            # Prepare metadata
            base_metadata = additional_metadata or {}
            base_metadata["file_id"] = document_id
            base_metadata["file_path"] = file_path  # Store original path
            base_metadata["local_file"] = True  # Flag to indicate this is a local file

            # Create document record in database first
            try:
                document = db_service.create_document(
                    filename=filename,
                    content_type=content_type,
                    storage_path=None,  # No MinIO storage path for local files
                    metadata={"local_path": file_path, **base_metadata},
                )
                document_id = document.id
                print(f"Created document record with ID: {document_id}")

                # Update file_id in metadata to ensure consistency
                base_metadata["file_id"] = document_id
            except Exception as db_error:
                print(f"Error creating document record: {str(db_error)}")
                return {
                    "file_path": file_path,
                    "status": "error",
                    "message": f"Database error: {str(db_error)}",
                }

            # Now process the document for embedding
            result = await embedding_service.process_document(
                file_content=file_content,
                filename=filename,
                content_type=content_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                base_metadata=base_metadata,
            )

            # Return success result
            return {
                "file_id": document_id,
                "file_path": file_path,
                "status": "success",
                "filename": filename,
                "chunks": result["chunks"],
                "vector_ids": result["vector_ids"],
            }

        except Exception as e:
            import traceback

            traceback.print_exc()
            return {"file_path": file_path, "status": "error", "message": str(e)}

    # Process all local files concurrently
    tasks = [process_single_local_file(file_path) for file_path in file_paths]
    results = await asyncio.gather(*tasks)

    # Organize results
    for result in results:
        if result.get("status") == "success":
            successful.append(result)
            total_chunks += len(result.get("chunks", []))
        else:
            failed.append(result)

    return MultiDocumentProcessResponse(
        successful=successful,
        failed=failed,
        total_processed=len(successful),
        total_chunks=total_chunks,
    )


@router.post("/embedding/batch", response_model=MultiDocumentProcessResponse)
async def batch_embedding_endpoint(
    request: MultiEmbeddingDocumentRequest, session_id: str = Depends(get_session_id)
):
    if request.additional_metadata is None:
        request.additional_metadata = {}
    request.additional_metadata["session_id"] = session_id
    """
    Process and embed multiple documents from file_ids
    """
    successful = []
    failed = []
    total_chunks = 0

    async def process_single_document(file_id):
        """Process a single document within the batch"""
        try:
            print(f"Processing file ID: {file_id}")

            # Get document details from database
            document = DatabaseService.get_document(file_id)
            if not document:
                print(f"Document not found: {file_id}")
                return {
                    "file_id": file_id,
                    "status": "error",
                    "message": "Document not found",
                }

            # Get file content
            try:
                file_content = storage_service.get_file_content(
                    document["storage_path"]
                )
                if not file_content:
                    print(f"File content not found: {document['storage_path']}")
                    return {
                        "file_id": file_id,
                        "status": "error",
                        "message": "File content not found",
                    }
            except Exception as storage_error:
                print(f"Storage error for file {file_id}: {str(storage_error)}")
                return {
                    "file_id": file_id,
                    "status": "error",
                    "message": f"Storage error: {str(storage_error)}",
                }

            # Process document with provided parameters
            chunk_size = (
                request.chunk_size
                if request.chunk_size is not None
                else DEFAULT_CHUNK_SIZE
            )
            chunk_overlap = (
                request.chunk_overlap
                if request.chunk_overlap is not None
                else DEFAULT_CHUNK_OVERLAP
            )

            # Prepare metadata
            base_metadata = request.additional_metadata or {}
            base_metadata["file_id"] = document["id"]

            # Process the document
            result = await embedding_service.process_document(
                file_content=file_content,
                filename=document["filename"],
                content_type=document["content_type"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                base_metadata=base_metadata,
            )

            # Return success result
            return {
                "file_id": file_id,
                "status": "success",
                "filename": document["filename"],
                "vector_ids": result["vector_ids"],
            }

        except Exception as e:
            print(f"Error processing file {file_id}: {str(e)}")
            import traceback

            traceback.print_exc()
            return {"file_id": file_id, "status": "error", "message": str(e)}

    # Process all documents concurrently
    tasks = [process_single_document(file_id) for file_id in request.file_ids]
    results = await asyncio.gather(*tasks)

    # Organize results
    for result in results:
        if result.get("status") == "success":
            successful.append(result)
            total_chunks += len(result.get("vector_ids", []))
        else:
            failed.append(result)

    return MultiDocumentProcessResponse(
        successful=successful,
        failed=failed,
        total_processed=len(successful),
        total_chunks=total_chunks,
    )


@router.post("/search", response_model=SearchResponse)
async def search_endpoint(
    request: SearchRequest, session_id: str = Depends(get_session_id)
):
    """
    Search for similar text based on semantic similarity
    Supports multiple filter conditions in filter_metadata
    """
    try:
        # Generate embedding for query
        query_embedding = embedding_service.embed_texts([request.query])[0]

        # Merge active filter with user-provided filter_metadata
        filter_conditions = request.filter_metadata or {}
        filter_conditions["active"] = (True,)
        filter_conditions["session_id"] = session_id
        print(f"Qdrant filter_conditions: {filter_conditions}")

        # Search vectors with filter conditions
        search_results = vector_db_service.search_vectors(
            query_vector=query_embedding,
            limit=request.limit,
            filter_conditions=filter_conditions,
        )
        print(f"Qdrant search results: {len(search_results)} items")
        for result in search_results:
            print(
                f"Qdrant result: id={result['id']}, file_id={result['metadata'].get('file_id')}, active={result['metadata'].get('active')}"
            )

        # Format results
        results = []
        for result in search_results:
            text = result.get("metadata", {}).get("text", "")
            metadata = {k: v for k, v in result["metadata"].items() if k != "text"}
            results.append(
                ChunkSearchResult(
                    id=result["id"], text=text, score=result["score"], metadata=metadata
                )
            )

        return SearchResponse(results=results)
    except Exception as e:
        print(f"Error in search_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")


@router.delete("/documents/batch")
async def delete_multiple_documents(
    document_ids: List[str] = Body(..., embed=True),
    session_id: str = Depends(get_session_id),
):
    """
    Delete multiple documents at once from all services:
    - PostgreSQL database
    - Vector database (Qdrant)
    - MinIO storage
    """
    results = {"successful": [], "failed": [], "total_deleted": 0}

    for file_id in document_ids:
        try:
            # Get document info before deleting
            doc_info = DatabaseService.get_document(file_id)
            if not doc_info:
                results["failed"].append({"id": file_id, "error": "Document not found"})
                continue

            # Check session ownership
            if doc_info.get("metadata", {}).get("session_id") != session_id:
                results["failed"].append({"id": file_id, "error": "Access denied"})
                continue

            # Get storage path
            storage_path = doc_info.get("storage_path")

            # Delete vectors from Qdrant
            vector_success = vector_db_service.delete_vectors_by_filter(
                {"file_id": file_id}
            )

            # Delete from MinIO if storage path exists
            minio_success = True
            if storage_path:
                try:
                    minio_success = storage_service.delete_file(storage_path)
                except Exception as e:
                    minio_success = False
                    print(f"Error deleting document {file_id} from MinIO: {e}")

            # Delete from PostgreSQL
            db_success = DatabaseService.delete_document(file_id)

            # Check if all operations were successful
            if vector_success and db_success and minio_success:
                results["successful"].append(
                    {"id": file_id, "message": "Successfully deleted"}
                )
                results["total_deleted"] += 1
            else:
                # Report partial success
                failures = []
                if not vector_success:
                    failures.append("vector database")
                if not db_success:
                    failures.append("database")
                if not minio_success:
                    failures.append("storage")

                results["failed"].append(
                    {
                        "id": file_id,
                        "error": f"Partially deleted. Failed in: {', '.join(failures)}",
                    }
                )

        except Exception as e:
            results["failed"].append({"id": file_id, "error": str(e)})

    return results


@router.delete("/documents/local/batch")
async def delete_local_documents(document_ids: List[str] = Body(..., embed=True)):
    """
    Delete multiple local document embeddings:
    - PostgreSQL database
    - Vector database (Qdrant)
    """
    results = {"successful": [], "failed": [], "total_deleted": 0}

    for file_id in document_ids:
        try:
            # Get document info before deleting
            doc_info = DatabaseService.get_document(file_id)
            if not doc_info:
                results["failed"].append({"id": file_id, "error": "Document not found"})
                continue

            # Check if it's a local document
            metadata = doc_info.get("metadata", {})
            if not metadata.get("local_file", False):
                results["failed"].append(
                    {
                        "id": file_id,
                        "error": "Not a local document. Use regular delete endpoint.",
                    }
                )
                continue

            # Delete vectors from Qdrant
            vector_success = vector_db_service.delete_vectors_by_filter(
                {"file_id": file_id}
            )

            # Delete from PostgreSQL
            db_success = DatabaseService.delete_document(file_id)

            # Check if all operations were successful
            if vector_success and db_success:
                results["successful"].append(
                    {"id": file_id, "message": "Successfully deleted"}
                )
                results["total_deleted"] += 1
            else:
                # Report partial success
                failures = []
                if not vector_success:
                    failures.append("vector database")
                if not db_success:
                    failures.append("database")

                results["failed"].append(
                    {
                        "id": file_id,
                        "error": f"Partially deleted. Failed in: {', '.join(failures)}",
                    }
                )

        except Exception as e:
            results["failed"].append({"id": file_id, "error": str(e)})

    return results


@router.post("/documents/{document_id}/toggle-status")
async def toggle_document_status(
    document_id: str,
    active: bool = Body(...),
    session_id: str = Depends(get_session_id),
):
    """
    Toggle the active status of a document in both PostgreSQL and Qdrant
    """
    try:
        document = DatabaseService.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Check session ownership
        if document.get("metadata", {}).get("session_id") != session_id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Update metadata
        metadata = document.get("metadata", {}) or {}
        metadata["active"] = active

        # Update in PostgreSQL
        db_success = DatabaseService.update_document_metadata(document_id, metadata)
        print(f"PostgreSQL update success: {db_success}")

        # Update in Qdrant
        try:
            vector_operation = vector_db_service.update_vectors_metadata(
                filter_conditions={"file_id": document_id},
                metadata_update={"active": active},
            )
            print(f"Vector database update result: {vector_operation}")
        except Exception as ve:
            print(f"Error in vector database operation: {str(ve)}")
            vector_operation = False

        if vector_operation and db_success:
            status = "enabled" if active else "disabled"
            return {"success": True, "message": f"Document {status} successfully"}
        else:
            failures = []
            if not vector_operation:
                failures.append("vector database")
            if not db_success:
                failures.append("database")
            print(f"Partial update failed in: {', '.join(failures)}")
            return {
                "success": False,
                "error": f"Partial update. Failed in: {', '.join(failures)}",
            }
    except Exception as e:
        print(f"Overall error in toggle_document_status: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error updating document: {str(e)}"
        )


@router.get("/documents", response_model=DocumentListResponse)
async def get_documents(limit: int = Query(100, ge=1), offset: int = Query(0, ge=0)):
    """
    Get a list of all documents in the system
    """
    try:
        documents = DatabaseService.get_documents(limit=limit, offset=offset)
        total = DatabaseService.count_documents()

        return DocumentListResponse(
            documents=documents, total=total, limit=limit, offset=offset
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving documents: {str(e)}"
        )


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """
    Get a specific document by ID
    """
    try:
        document = DatabaseService.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return DocumentResponse(**document)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving document: {str(e)}"
        )
