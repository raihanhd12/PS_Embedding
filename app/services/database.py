import datetime
import os
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

import app.utils.config as config

# Create SQLAlchemy engine with fallback to SQLite for development
try:
    # Try to connect to PostgreSQL
    engine = create_engine(config.DB_URL)
    print("Using PostgreSQL database")
except Exception as e:
    # Fallback to SQLite for development/testing
    print(f"PostgreSQL connection failed ({str(e)})")
    print("Using SQLite for development")
    sqlite_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "embeddings.db"
    )
    engine = create_engine(f"sqlite:///{sqlite_path}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Define SQLAlchemy models
class Document(Base):
    """Document metadata model"""

    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    storage_path = Column(String, nullable=True)  # Path in MinIO
    # Change from func.now to datetime.datetime.now
    created_at = Column(DateTime, default=datetime.datetime.now)
    chunks = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )
    doc_metadata = Column(JSON, nullable=True)


class DocumentChunk(Base):
    """Document chunk model"""

    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    embedding_id = Column(String, nullable=True)  # ID in vector DB
    document = relationship("Document", back_populates="chunks")
    chunk_metadata = Column(JSON, nullable=True)


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
            print(
                "The application might not function correctly without database access"
            )

    @staticmethod
    def get_session():
        """Get a database session"""
        return SessionLocal()

    @staticmethod
    def create_document(
        filename: str,
        content_type: str,
        storage_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        try:
            with SessionLocal() as session:
                document = Document(
                    id=str(uuid.uuid4()),
                    filename=filename,
                    content_type=content_type,
                    storage_path=storage_path,
                    doc_metadata=metadata
                    or {},  # Use doc_metadata field name from model
                )
                print(f"Mencoba menyimpan dokumen dengan ID: {document.id}")
                session.add(document)
                session.commit()
                session.refresh(document)
                print(f"Dokumen berhasil disimpan ke database: {document.id}")
                return document
        except Exception as e:
            print(f"ERROR SAAT MEMBUAT DOKUMEN: {str(e)}")
            print(f"Tipe exception: {type(e).__name__}")
            # Raise exception instead of returning mock object
            raise e

    @staticmethod
    def create_document_chunk(
        document_id: str,
        chunk_index: int,
        text: str,
        embedding_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DocumentChunk:
        """
        Create a new document chunk record

        Args:
            document_id: Parent document ID
            chunk_index: Chunk index in document
            text: Chunk text content
            embedding_id: Optional vector DB embedding ID
            metadata: Optional chunk metadata

        Returns:
            DocumentChunk: Created chunk
        """
        try:
            with SessionLocal() as session:
                chunk = DocumentChunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    chunk_index=chunk_index,
                    text=text,
                    embedding_id=embedding_id,
                    metadata=metadata or {},
                )
                session.add(chunk)
                session.commit()
                session.refresh(chunk)
                return chunk
        except Exception as e:
            print(f"Error creating document chunk: {e}")
            # Return a mock chunk with ID for fallback functionality
            return DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                chunk_index=chunk_index,
                text=text,
                embedding_id=embedding_id,
                chunk_metadata=metadata or {},
            )

    @staticmethod
    def get_document(document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document by ID

        Args:
            document_id: Document ID

        Returns:
            Dict: Document data or None if not found
        """
        try:
            with SessionLocal() as session:
                document = (
                    session.query(Document).filter(Document.id == document_id).first()
                )
                if not document:
                    return None

                return {
                    "id": document.id,
                    "filename": document.filename,
                    "content_type": document.content_type,
                    "storage_path": document.storage_path,
                    "created_at": document.created_at,
                    "metadata": document.doc_metadata,
                }
        except Exception as e:
            print(f"Error getting document: {e}")
            return None

    @staticmethod
    def get_documents(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get list of documents

        Args:
            limit: Maximum number of documents
            offset: Query offset

        Returns:
            List[Dict]: List of document data
        """
        try:
            with SessionLocal() as session:
                documents = (
                    session.query(Document)
                    .order_by(Document.created_at.desc())
                    .offset(offset)
                    .limit(limit)
                    .all()
                )

                return [
                    {
                        "id": doc.id,
                        "filename": doc.filename,
                        "content_type": doc.content_type,
                        "storage_path": doc.storage_path,
                        "created_at": doc.created_at,
                        "metadata": doc.doc_metadata,
                    }
                    for doc in documents
                ]
        except Exception as e:
            print(f"Error getting documents: {e}")
            return []

    @staticmethod
    def get_document_chunks(document_id: str) -> List[Dict[str, Any]]:
        """
        Get chunks for a document

        Args:
            document_id: Document ID

        Returns:
            List[Dict]: List of chunk data
        """
        try:
            with SessionLocal() as session:
                chunks = (
                    session.query(DocumentChunk)
                    .filter(DocumentChunk.document_id == document_id)
                    .order_by(DocumentChunk.chunk_index)
                    .all()
                )

                return [
                    {
                        "id": chunk.id,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "embedding_id": chunk.embedding_id,
                        "metadata": chunk.chunk_metadata,
                    }
                    for chunk in chunks
                ]
        except Exception as e:
            print(f"Error getting document chunks: {e}")
            return []

    @staticmethod
    def delete_document(document_id: str) -> bool:
        """
        Delete document and all its chunks

        Args:
            document_id: Document ID

        Returns:
            bool: True if successful
        """
        try:
            with SessionLocal() as session:
                document = (
                    session.query(Document).filter(Document.id == document_id).first()
                )
                if not document:
                    return False

                session.delete(document)
                session.commit()
                return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False

    @staticmethod
    def update_document_metadata(document_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update document metadata

        Args:
            document_id: Document ID
            metadata: Updated metadata dictionary

        Returns:
            bool: True if successful
        """
        try:
            with SessionLocal() as session:
                document = (
                    session.query(Document).filter(Document.id == document_id).first()
                )
                if not document:
                    return False

                # Update metadata - preserve existing data and add/overwrite new fields
                current_metadata = document.doc_metadata or {}
                updated_metadata = {**current_metadata, **metadata}
                document.doc_metadata = updated_metadata

                session.commit()
                return True
        except Exception as e:
            print(f"Error updating document metadata: {e}")
            return False

    @staticmethod
    def update_chunk_embedding(chunk_id: str, embedding_id: str) -> bool:
        """
        Update chunk with embedding ID

        Args:
            chunk_id: Chunk ID
            embedding_id: Vector DB embedding ID

        Returns:
            bool: True if successful
        """
        try:
            with SessionLocal() as session:
                chunk = (
                    session.query(DocumentChunk)
                    .filter(DocumentChunk.id == chunk_id)
                    .first()
                )
                if not chunk:
                    return False

                chunk.embedding_id = embedding_id
                session.commit()
                return True
        except Exception as e:
            print(f"Error updating chunk embedding: {e}")
            return False

    @staticmethod
    def count_documents() -> int:
        """
        Count total number of documents in the database

        Returns:
            int: Total document count
        """
        try:
            with SessionLocal() as session:
                return session.query(Document).count()
        except Exception as e:
            print(f"Error counting documents: {e}")
            return 0
