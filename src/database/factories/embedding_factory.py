import datetime
import uuid

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Document(Base):
    """Document metadata model"""

    __tablename__ = "documents"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    storage_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.now)
    chunks = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )
    images = relationship(
        "DocumentImage", back_populates="document", cascade="all, delete-orphan"
    )
    doc_metadata = Column(JSON, nullable=True)


class DocumentChunk(Base):
    """Document chunk model"""

    __tablename__ = "document_chunks"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    embedding_id = Column(String, nullable=True)
    page_number = Column(Integer, nullable=True)
    document = relationship("Document", back_populates="chunks")
    chunk_metadata = Column(JSON, nullable=True)
    related_images = Column(JSON, nullable=True)


class DocumentImage(Base):
    """Document image model"""

    __tablename__ = "document_images"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"))
    page_number = Column(Integer, nullable=False)
    image_index = Column(Integer, nullable=False)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    format = Column(String, nullable=True)
    storage_path = Column(String, nullable=True)
    ocr_text = Column(Text, nullable=True)
    document = relationship("Document", back_populates="images")
    image_metadata = Column(JSON, nullable=True)
