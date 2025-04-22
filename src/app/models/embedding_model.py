from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentImagePydantic(BaseModel):
    """Pydantic model for document image data."""

    id: Optional[str] = None
    document_id: str
    page_number: int
    image_index: int
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    storage_path: Optional[str] = None
    ocr_text: Optional[str] = None
    image_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        from_attributes = True  # Allow conversion from SQLAlchemy objects


class DocumentChunkPydantic(BaseModel):
    """Pydantic model for document chunk data."""

    id: Optional[str] = None
    document_id: str
    chunk_index: int
    text: str
    page_number: Optional[int] = None
    embedding_id: Optional[str] = None
    chunk_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    related_images: Optional[List[str]] = None

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class DocumentPydantic(BaseModel):
    """Pydantic model for document metadata."""

    id: Optional[str] = None
    filename: str
    content_type: str
    storage_path: Optional[str] = None
    created_at: Optional[datetime] = Field(default_factory=lambda: datetime.utcnow())
    doc_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    chunks: Optional[List[DocumentChunkPydantic]] = None
    images: Optional[List[DocumentImagePydantic]] = None

    class Config:
        """Pydantic configuration."""

        from_attributes = True
