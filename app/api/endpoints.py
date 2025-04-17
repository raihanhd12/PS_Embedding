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

from app.models.schemas import (
    ChunkSearchResult,
    DocumentListResponse,
    DocumentResponse,
    MultiDocumentProcessResponse,
    MultiDocumentUploadResponse,
    MultiEmbeddingDocumentRequest,
    SearchRequest,
    SearchResponse,
)
from app.services.database import DatabaseService
from app.services.elasticsearch import ElasticsearchService
from app.services.embedding import embed_texts
from app.services.storage import StorageService
from app.services.text_processor import process_document
from app.services.vector_db import (
    delete_vectors_by_filter,
    search_vectors,
    update_vectors_metadata,
)
from app.utils.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from app.utils.security import validate_api_key

# Create API router
router = APIRouter(prefix="/api", dependencies=[Depends(validate_api_key)])


@router.post("/upload/batch", response_model=MultiDocumentUploadResponse)
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(None),
):
    """Upload multiple documents in a single request"""
    successful = []
    failed = []

    # Parse metadata if provided
    metadata_dict = json.loads(metadata) if metadata else {}
    metadata_dict["active"] = True  # Set default active status

    # Create a storage service instance to reuse
    storage_service = StorageService()

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
            db_service = DatabaseService()
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
    _: str = Depends(validate_api_key),
):
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

            # Process the document directly
            result = await process_document(
                file_content=file_content,
                filename=filename,
                content_type=content_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                base_metadata=base_metadata,
            )

            # Optionally create a document record to track this in the database
            # This is optional since we're not using MinIO, but helps with tracking
            try:
                db_service = DatabaseService()
                document = db_service.create_document(
                    filename=filename,
                    content_type=content_type,
                    storage_path=None,  # No MinIO storage path
                    metadata={"local_path": file_path, **base_metadata},
                )
                document_id = document.id
            except Exception as db_error:
                print(f"Warning: Could not create document record: {str(db_error)}")
                # Continue even if document record creation fails

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
    import asyncio

    tasks = [process_single_local_file(file_path) for file_path in file_paths]
    results = await asyncio.gather(*tasks)

    # Organize results
    for result in results:
        if result.get("status") == "success":
            successful.append(result)
            total_chunks += result.get("chunks", 0)
        else:
            failed.append(result)

    return MultiDocumentProcessResponse(
        successful=successful,
        failed=failed,
        total_processed=len(successful),
        total_chunks=total_chunks,
    )


@router.post("/embedding/batch", response_model=MultiDocumentProcessResponse)
async def batch_embedding_endpoint(request: MultiEmbeddingDocumentRequest):
    """
    Process and embed multiple documents from file_ids
    """
    successful = []
    failed = []
    total_chunks = 0

    # Create a storage service to reuse
    storage_service = StorageService()

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
            result = await process_document(
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
            total_chunks += len(result.get("chunks", []))
        else:
            failed.append(result)

    return MultiDocumentProcessResponse(
        successful=successful,
        failed=failed,
        total_processed=len(successful),
        total_chunks=total_chunks,
    )


@router.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    """
    Search for similar text based on semantic similarity
    Supports multiple filter conditions in filter_metadata
    """
    try:
        # Generate embedding for query
        query_embedding = embed_texts([request.query])[0]

        # Merge active filter with user-provided filter_metadata
        filter_conditions = request.filter_metadata or {}
        filter_conditions["active"] = True
        print(f"Qdrant filter_conditions: {filter_conditions}")

        # Search vectors with filter conditions
        search_results = search_vectors(
            query_vector=query_embedding,
            limit=request.limit,
            filter_conditions=filter_conditions,
        )
        print(f"Qdrant search results: {len(search_results)} items")
        for result in search_results:
            print(
                f"Qdrant result: id={result['id']}, file_id={result['metadata'].get('file_id')}, active={result['metadata'].get('active')}"
            )

        # Perform hybrid search with the same filter
        es_service = ElasticsearchService()
        hybrid_results = es_service.hybrid_search(
            query=request.query,
            vector_results=search_results,
            vector_weight=0.7,
            keyword_weight=0.3,
            limit=request.limit,
            filters={"metadata.active": True},
        )
        print(f"Hybrid search results: {len(hybrid_results)} items")
        for result in hybrid_results:
            print(
                f"Hybrid result: id={result['id']}, file_id={result['metadata'].get('file_id')}, active={result['metadata'].get('active')}"
            )

        # Format results
        results = []
        for result in hybrid_results:
            text = result.get("text", result["metadata"].get("text", ""))
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


# router delete all documents
@router.delete("/documents/batch")
async def delete_multiple_documents(document_ids: List[str] = Body(..., embed=True)):
    """
    Delete multiple documents at once from all services:
    - PostgreSQL database
    - Vector database (Qdrant)
    - Elasticsearch
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

            # Get storage path
            storage_path = doc_info.get("storage_path")

            # Delete vectors from Qdrant
            vector_success = delete_vectors_by_filter({"file_id": file_id})

            # Delete from Elasticsearch if available
            es_success = True
            try:
                es_service = ElasticsearchService()
                es_success = es_service.delete_by_query(
                    {"query": {"term": {"file_id": file_id}}}
                )
            except Exception as e:
                es_success = False
                print(f"Error deleting document {file_id} from Elasticsearch: {e}")

            # Delete from MinIO if storage path exists
            minio_success = True
            if storage_path:
                try:
                    storage_service = StorageService()
                    minio_success = storage_service.delete_file(storage_path)
                except Exception as e:
                    minio_success = False
                    print(f"Error deleting document {file_id} from MinIO: {e}")

            # Delete from PostgreSQL
            db_success = DatabaseService.delete_document(file_id)

            # Check if all operations were successful
            if vector_success and db_success and es_success and minio_success:
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
                if not es_success:
                    failures.append("Elasticsearch")
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
    - Elasticsearch
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
            vector_success = delete_vectors_by_filter({"file_id": file_id})

            # Delete from Elasticsearch if available
            es_success = True
            try:
                es_service = ElasticsearchService()
                es_success = es_service.delete_by_query(
                    {"query": {"term": {"file_id": file_id}}}
                )
            except Exception as e:
                es_success = False
                print(f"Error deleting document {file_id} from Elasticsearch: {e}")

            # Delete from PostgreSQL
            db_success = DatabaseService.delete_document(file_id)

            # Check if all operations were successful
            if vector_success and db_success and es_success:
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
                if not es_success:
                    failures.append("Elasticsearch")

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
):
    try:
        document = DatabaseService.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        metadata = document.get("metadata", {}) or {}
        metadata["active"] = active
        db_success = DatabaseService.update_document_metadata(document_id, metadata)
        print(f"PostgreSQL update success: {db_success}")
        try:
            vector_operation = update_vectors_metadata(
                filter_conditions={"file_id": document_id},
                metadata_update={"active": active},
            )
            print(f"Vector database update result: {vector_operation}")
        except Exception as ve:
            print(f"Error in vector database operation: {str(ve)}")
            vector_operation = False
        es_success = True
        try:
            es_service = ElasticsearchService()
            search_response = es_service.search(
                query="",
                filters={"file_id": document_id},
                limit=1,
            )
            print(
                f"Elasticsearch search found {len(search_response)} documents for file_id: {document_id}"
            )
            if not search_response:
                print(f"No document found in Elasticsearch with file_id: {document_id}")
                es_success = False
            else:
                es_update_query = {
                    "script": {
                        "source": "ctx._source.metadata.active = params.active",
                        "params": {"active": active},
                    },
                    "query": {"term": {"file_id": document_id}},
                }
                es_success = es_service.update_by_query(es_update_query)
                print(f"Elasticsearch update success: {es_success}")
                if not es_success:
                    print(f"Elasticsearch update failed for document {document_id}")
        except Exception as es_e:
            print(f"Elasticsearch operation failed: {str(es_e)}")
            es_success = False
        if vector_operation and db_success:
            status = "enabled" if active else "disabled"
            return {"success": True, "message": f"Document {status} successfully"}
        else:
            failures = []
            if not vector_operation:
                failures.append("vector database")
            if not db_success:
                failures.append("database")
            if not es_success:
                failures.append("Elasticsearch")
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
        # Assuming this method exists or needs to be created
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
