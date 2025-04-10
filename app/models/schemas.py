"""
Pydantic models for request and response schemas
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.utils.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

# Request model
class SearchRequest(BaseModel):
    """Request model for vector search"""
    query: str = Field(..., description="Search query text")
    limit: int = Field(5, description="Maximum number of results to return")
    filter_metadata: Optional[Dict[str, Any]] = Field(
        None, description="Optional metadata filters with multiple conditions")


class EmbeddingDocumentRequest(BaseModel):
    """Request model for direct document embedding"""
    file_id: str = Field(..., description="Document ID to process and embed")
    chunk_size: Optional[int] = DEFAULT_CHUNK_SIZE
    chunk_overlap: Optional[int] = DEFAULT_CHUNK_OVERLAP
    additional_metadata: Optional[Dict[str, Any]] = None


# Response models


class ChunkSearchResult(BaseModel):
    """Model for a single search result"""
    id: str = Field(..., description="Vector ID")
    text: str = Field(..., description="Text content")
    score: float = Field(..., description="Similarity score")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata")


class SearchResponse(BaseModel):
    """Response model for search results"""
    results: List[ChunkSearchResult] = Field(..., description="Search results")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    file_id: str = Field(..., description="Unique file ID")
    filename: str = Field(..., description="Original filename")
    storage_path: str = Field(..., description="Storage path in MinIO")
    content_type: str = Field(..., description="Content type")


class DocumentProcessResponse(BaseModel):
    """Response model for document processing"""
    filename: str = Field(..., description="Original filename")
    chunks: int = Field(..., description="Number of chunks created")
    vector_ids: List[str] = Field(...,
                                  description="Vector IDs for stored chunks")
    file_id: str = Field(..., description="Unique file ID")


# New response models for GET endpoints

class DocumentResponse(BaseModel):
    """Response model for a single document"""
    id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="Content type")
    storage_path: Optional[str] = Field(None, description="Storage path")
    created_at: Optional[datetime] = Field(
        None, description="Creation timestamp")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Document metadata")


class DocumentListResponse(BaseModel):
    """Response model for listing documents"""
    documents: List[DocumentResponse] = Field(...,
                                              description="List of documents")
    total: int = Field(..., description="Total number of documents")
    limit: int = Field(..., description="Limit used for pagination")
    offset: int = Field(..., description="Offset used for pagination")


class MultiDocumentUploadResponse(BaseModel):
    """Response for multi-document upload"""
    successful: List[Dict[str, Any]]
    failed: List[Dict[str, Any]]
    total_uploaded: int


class MultiEmbeddingDocumentRequest(BaseModel):
    """Request model for embedding multiple documents"""
    file_ids: List[str]
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    additional_metadata: Optional[Dict[str, Any]] = None


class MultiDocumentProcessResponse(BaseModel):
    """Response model for multi-document embedding process"""
    successful: List[Dict[str, Any]]
    failed: List[Dict[str, Any]]
    total_processed: int
    total_chunks: int
