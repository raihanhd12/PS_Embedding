"""
Fixed DatabaseService class with proper method decorators
"""

import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.config.env as env
from src.app.models.embedding_model import (
    DocumentChunkPydantic,
    DocumentImagePydantic,
    DocumentPydantic,
)
from src.database.factories.embedding_factory import (
    Base,
    Document,
    DocumentChunk,
    DocumentImage,
)

# Database connection setup
engine = create_engine(env.DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class DatabaseException(Exception):
    """Custom exception for database-related errors."""

    pass


class DatabaseService:
    """Service for database operations"""

    @staticmethod
    def init_db():
        """Initialize database tables"""
        try:
            Base.metadata.create_all(bind=engine)
            print("Database tables initialized")
        except Exception as e:
            print(f"Error initializing database tables: {e}")

    @staticmethod
    @contextmanager
    def get_session():
        """Get a database session as a context manager."""
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    @classmethod
    def create_document(cls, document: DocumentPydantic) -> DocumentPydantic:
        """Create a new document record.

        Args:
            document: Pydantic model containing document data.

        Returns:
            Created document as Pydantic model.
        """
        try:
            with cls.get_session() as session:
                db_document = Document(
                    id=str(uuid.uuid4()),
                    filename=document.filename,
                    content_type=document.content_type,
                    storage_path=document.storage_path,
                    doc_metadata=document.doc_metadata or {},
                )
                session.add(db_document)
                session.commit()
                session.refresh(db_document)
                return DocumentPydantic.from_orm(db_document)
        except Exception as e:
            raise DatabaseException(f"Error creating document: {str(e)}")

    @classmethod  # Changed from staticmethod to classmethod
    def create_document_chunk(
        cls, chunk: DocumentChunkPydantic
    ) -> DocumentChunkPydantic:
        try:
            with cls.get_session() as session:
                db_chunk = DocumentChunk(
                    id=str(uuid.uuid4()),
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    page_number=chunk.page_number,
                    embedding_id=chunk.embedding_id,
                    chunk_metadata=chunk.chunk_metadata or {},
                    related_images=chunk.related_images,
                )
                session.add(db_chunk)
                session.commit()
                session.refresh(db_chunk)
                return DocumentChunkPydantic.from_orm(db_chunk)
        except Exception as e:
            raise DatabaseException(f"Error creating document chunk: {str(e)}")

    @classmethod  # Changed from staticmethod to classmethod
    def create_document_image(
        cls, image: DocumentImagePydantic
    ) -> DocumentImagePydantic:
        """Create a new document image record.

        Args:
            image: Pydantic model containing image data.

        Returns:
            Created image as Pydantic model.
        """
        try:
            with cls.get_session() as session:
                db_image = DocumentImage(
                    id=str(uuid.uuid4()),
                    document_id=image.document_id,
                    page_number=image.page_number,
                    image_index=image.image_index,
                    width=image.width,
                    height=image.height,
                    format=image.format,
                    storage_path=image.storage_path,
                    ocr_text=image.ocr_text,
                    image_metadata=image.image_metadata or {},
                )
                session.add(db_image)
                session.commit()
                session.refresh(db_image)
                return DocumentImagePydantic.from_orm(db_image)
        except Exception as e:
            raise DatabaseException(f"Error creating document image: {str(e)}")

    @classmethod  # Changed from staticmethod to classmethod
    def get_document(cls, document_id: str) -> Optional[DocumentPydantic]:
        """Get document by ID.

        Args:
            document_id: Document ID.

        Returns:
            Document as Pydantic model or None if not found.
        """
        try:
            with cls.get_session() as session:
                db_document = (
                    session.query(Document).filter(Document.id == document_id).first()
                )
                if not db_document:
                    return None
                return DocumentPydantic.from_orm(db_document)
        except Exception as e:
            raise DatabaseException(f"Error getting document: {str(e)}")

    @classmethod
    def get_documents(cls, limit: int = 100, offset: int = 0) -> List[DocumentPydantic]:
        """Get list of documents.

        Args:
            limit: Maximum number of documents.
            offset: Query offset.

        Returns:
            List of documents as Pydantic models.
        """
        try:
            with cls.get_session() as session:
                db_documents = (
                    session.query(Document)
                    .order_by(Document.created_at.desc())
                    .offset(offset)
                    .limit(limit)
                    .all()
                )
                return [DocumentPydantic.from_orm(doc) for doc in db_documents]
        except Exception as e:
            raise DatabaseException(f"Error getting documents: {str(e)}")

    @classmethod
    def get_document_chunks(cls, document_id: str) -> List[DocumentChunkPydantic]:
        """Get chunks for a document.

        Args:
            document_id: Document ID.

        Returns:
            List of chunks as Pydantic models.
        """
        try:
            with cls.get_session() as session:
                db_chunks = (
                    session.query(DocumentChunk)
                    .filter(DocumentChunk.document_id == document_id)
                    .order_by(DocumentChunk.chunk_index)
                    .all()
                )
                return [DocumentChunkPydantic.from_orm(chunk) for chunk in db_chunks]
        except Exception as e:
            raise DatabaseException(f"Error getting document chunks: {str(e)}")

    @classmethod
    def get_document_images(cls, document_id: str) -> List[DocumentImagePydantic]:
        """Get images for a document.

        Args:
            document_id: Document ID.

        Returns:
            List of images as Pydantic models.
        """
        try:
            with cls.get_session() as session:
                db_images = (
                    session.query(DocumentImage)
                    .filter(DocumentImage.document_id == document_id)
                    .order_by(DocumentImage.page_number, DocumentImage.image_index)
                    .all()
                )
                return [DocumentImagePydantic.from_orm(img) for img in db_images]
        except Exception as e:
            raise DatabaseException(f"Error getting document images: {str(e)}")

    @classmethod
    def delete_document(cls, document_id: str) -> bool:
        """Delete document and all its chunks and images.

        Args:
            document_id: Document ID.

        Returns:
            True if successful, False if document not found.
        """
        try:
            with cls.get_session() as session:
                db_document = (
                    session.query(Document).filter(Document.id == document_id).first()
                )
                if not db_document:
                    return False
                session.delete(db_document)
                session.commit()
                return True
        except Exception as e:
            raise DatabaseException(f"Error deleting document: {str(e)}")

    @classmethod
    def update_chunk_embedding(cls, chunk_id: str, embedding_id: str) -> bool:
        """Update chunk with embedding ID.

        Args:
            chunk_id: Chunk ID.
            embedding_id: Vector DB embedding ID.

        Returns:
            True if successful, False if chunk not found.
        """
        try:
            with cls.get_session() as session:
                db_chunk = (
                    session.query(DocumentChunk)
                    .filter(DocumentChunk.id == chunk_id)
                    .first()
                )
                if not db_chunk:
                    return False
                db_chunk.embedding_id = embedding_id
                session.commit()
                return True
        except Exception as e:
            raise DatabaseException(f"Error updating chunk embedding: {str(e)}")

    @classmethod
    def count_documents(cls) -> int:
        """Count total number of documents in the database.

        Returns:
            Total document count.
        """
        try:
            with cls.get_session() as session:
                return session.query(Document).count()
        except Exception as e:
            raise DatabaseException(f"Error counting documents: {str(e)}")

    @classmethod
    def update_document_metadata(
        cls, document_id: str, metadata: Dict[str, Any]
    ) -> bool:
        """Update document metadata.

        Args:
            document_id: Document ID.
            metadata: Updated metadata dictionary.

        Returns:
            True if successful, False if document not found.
        """
        try:
            with cls.get_session() as session:
                db_document = (
                    session.query(Document).filter(Document.id == document_id).first()
                )
                if not db_document:
                    return False
                current_metadata = db_document.doc_metadata or {}
                updated_metadata = {**current_metadata, **metadata}
                db_document.doc_metadata = updated_metadata
                session.commit()
                return True
        except Exception as e:
            raise DatabaseException(f"Error updating document metadata: {str(e)}")
