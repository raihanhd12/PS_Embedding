"""
Pydantic models for request and response schemas
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Request models


class TextEmbeddingRequest(BaseModel):
    """Request model for text embedding"""
    texts: List[str] = Field(..., description="List of texts to embed")
    metadata: Optional[List[Dict[str, Any]]] = Field(
        None, description="Optional metadata for each text")
    store: bool = Field(
        False, description="Whether to store embeddings in vector DB")


class SearchRequest(BaseModel):
    """Request model for vector search"""
    query: str = Field(..., description="Search query text")
    limit: int = Field(5, description="Maximum number of results to return")
    filter_metadata: Optional[Dict[str, Any]] = Field(
        None, description="Optional metadata filters")

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


class EmbeddingResponse(BaseModel):
    """Response model for embedding results"""
    count: int = Field(..., description="Number of embeddings")
    dimension: int = Field(..., description="Embedding dimension")
    vector_ids: Optional[List[str]] = Field(
        None, description="Vector IDs if stored")
    embeddings: Optional[List[List[float]]] = Field(
        None, description="Raw embeddings if not stored")


class DocumentProcessResponse(BaseModel):
    """Response model for document processing"""
    filename: str = Field(..., description="Original filename")
    chunks: int = Field(..., description="Number of chunks created")
    vector_ids: List[str] = Field(...,
                                  description="Vector IDs for stored chunks")
    file_id: str = Field(..., description="Unique file ID")
