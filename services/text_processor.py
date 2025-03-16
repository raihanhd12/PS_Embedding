"""
Text processing service for extraction and chunking
"""
from typing import List, Dict, Any, Optional
import uuid

import config
from utils.file_utils import extract_text_from_file
from services.embedding import embed_texts, create_embeddings
from services.vector_db import store_vectors


def split_text_into_chunks(
    text: str,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    overlap: int = config.DEFAULT_CHUNK_OVERLAP
) -> List[str]:
    """
    Split text into overlapping chunks

    Args:
        text: Text to split
        chunk_size: Maximum chunk size in characters
        overlap: Overlap between chunks in characters

    Returns:
        List of text chunks
    """
    if not text:
        return []

    # Simple splitting by paragraphs first, then recombining
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If adding this paragraph exceeds chunk size, save current chunk and start new one
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap from end of previous chunk
            words = current_chunk.split()
            if len(words) > overlap:
                current_chunk = " ".join(words[-overlap:]) + "\n\n"
            else:
                current_chunk = ""

        current_chunk += para + "\n\n"

        # If current chunk is now too big, split it further
        while len(current_chunk) > chunk_size:
            chunks.append(current_chunk[:chunk_size].strip())
            # Keep overlap from end of previous chunk
            words = current_chunk[:chunk_size].split()
            if len(words) > overlap:
                current_chunk = " ".join(
                    words[-overlap:]) + "\n\n" + current_chunk[chunk_size:]
            else:
                current_chunk = current_chunk[chunk_size:]

    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


async def process_document(
    file_content: bytes,
    filename: str,
    content_type: str,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
    base_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process a document: extract text, chunk it, embed chunks, and store in vector DB

    Args:
        file_content: Binary file content
        filename: Original filename
        content_type: MIME content type
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        base_metadata: Optional base metadata to include with all chunks

    Returns:
        Dictionary with processing results
    """
    # Extract text
    text = extract_text_from_file(file_content, content_type)
    if not text:
        raise ValueError("Could not extract text from file")

    # Split into chunks
    chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("No text chunks generated")

    # Initialize metadata
    if base_metadata is None:
        base_metadata = {}

    # Add file info to metadata
    file_id = str(uuid.uuid4())
    base_metadata.update({
        "filename": filename,
        "content_type": content_type,
        "file_id": file_id
    })

    # Create metadata for each chunk
    metadata_list = []
    for i, chunk in enumerate(chunks):
        chunk_metadata = base_metadata.copy()
        chunk_metadata.update({
            "chunk_index": i,
            "text": chunk
        })
        metadata_list.append(chunk_metadata)

    # Generate embeddings
    embeddings = embed_texts(chunks)

    # Generate IDs and store in vector DB
    ids = [str(uuid.uuid4()) for _ in chunks]
    store_vectors(embeddings, metadata_list, ids)

    return {
        "filename": filename,
        "chunks": len(chunks),
        "vector_ids": ids,
        "file_id": file_id
    }
