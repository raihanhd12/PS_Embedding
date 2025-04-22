"""
API routes for the Embedding API with clean separation of routes and controllers
"""

from typing import List, Optional

from fastapi import APIRouter, Body, Depends, File, Form, Query, UploadFile

from src.app.controllers.embedding_controller import EmbeddingController
from src.app.schemas.embedding_schema import (
    DocumentListResponse,
    DocumentResponse,
    MultiDocumentProcessResponse,
    MultiDocumentUploadResponse,
    MultiEmbeddingDocumentRequest,
    SearchRequest,
    SearchResponse,
)
from src.config.security import validate_api_key
from src.config.session import get_session_id

# Create API router
router = APIRouter(prefix="/api/v1", dependencies=[Depends(validate_api_key)])

# Initialize controller
embedding_controller = EmbeddingController()

@router.post("/upload/batch", response_model=MultiDocumentUploadResponse)
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(None),
    session_id: str = Depends(get_session_id),
):
    """Upload multiple documents in a single request"""
    return await embedding_controller.upload_documents(files, metadata, session_id)


@router.post("/embedding/local", response_model=MultiDocumentProcessResponse)
async def local_file_embedding(
    file_paths: List[str] = Body(..., description="List of local file paths"),
    chunk_size: Optional[int] = Body(None, description="Chunk size for text splitting"),
    chunk_overlap: Optional[int] = Body(
        None, description="Chunk overlap for text splitting"
    ),
    additional_metadata: Optional[dict] = Body(
        None, description="Additional metadata for all files"
    ),
    session_id: str = Depends(get_session_id),
    _: str = Depends(validate_api_key),
):
    """Process and embed local files"""
    return await embedding_controller.local_file_embedding(
        file_paths, chunk_size, chunk_overlap, additional_metadata, session_id
    )


@router.post("/embedding/batch", response_model=MultiDocumentProcessResponse)
async def batch_embedding_endpoint(
    request: MultiEmbeddingDocumentRequest, session_id: str = Depends(get_session_id)
):
    """Process and embed multiple documents from file_ids"""
    return await embedding_controller.batch_embedding(request, session_id)


@router.post("/search", response_model=SearchResponse)
async def search_endpoint(
    request: SearchRequest, session_id: str = Depends(get_session_id)
):
    """
    Search for similar text based on semantic similarity
    Supports multiple filter conditions in filter_metadata
    """
    return await embedding_controller.search(request, session_id)


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
    return await embedding_controller.delete_documents(document_ids, session_id)


@router.delete("/documents/local/batch")
async def delete_local_documents(document_ids: List[str] = Body(..., embed=True)):
    """
    Delete multiple local document embeddings:
    - PostgreSQL database
    - Vector database (Qdrant)
    """
    return await embedding_controller.delete_local_documents(document_ids)


@router.post("/documents/{document_id}/toggle-status")
async def toggle_document_status(
    document_id: str,
    active: bool = Body(...),
    session_id: str = Depends(get_session_id),
):
    """
    Toggle the active status of a document in both PostgreSQL and Qdrant
    """
    return await embedding_controller.toggle_document_status(
        document_id, active, session_id
    )


@router.get("/documents", response_model=DocumentListResponse)
async def get_documents(limit: int = Query(100, ge=1), offset: int = Query(0, ge=0)):
    """
    Get a list of all documents in the system
    """
    return await embedding_controller.get_documents(limit, offset)


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """
    Get a specific document by ID
    """
    return await embedding_controller.get_document(document_id)
    return await embedding_controller.get_document(document_id)
