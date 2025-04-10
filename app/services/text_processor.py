"""
Text processing service for extraction and chunking
"""

import uuid
from typing import Any, Dict, List, Optional

import app.utils.config as config
from app.services.embedding import create_embeddings
from app.services.vector_db import store_vectors
from app.utils.file_utils import extract_text_from_file


def split_text_into_chunks(
    text: str,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    overlap: int = config.DEFAULT_CHUNK_OVERLAP,
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
    paragraphs = text.split("\n\n")
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
                current_chunk = (
                    " ".join(words[-overlap:]) + "\n\n" + current_chunk[chunk_size:]
                )
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
    base_metadata: Optional[Dict[str, Any]] = None,
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
    try:
        # Extract text
        text = extract_text_from_file(file_content, content_type)
        if not text or len(text.strip()) < 10:  # Require some minimum amount of text
            print(f"Warning: Insufficient text extracted from file: {filename}")
            raise ValueError("Insufficient text extracted from file")

        # Split into chunks
        chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
        if not chunks:
            raise ValueError("No text chunks generated")

        # Initialize metadata
        if base_metadata is None:
            base_metadata = {}

        # Get document ID from metadata if it exists
        document_id = base_metadata.get("file_id")
        if not document_id:
            print("Warning: No document ID provided in metadata")
            document_id = str(uuid.uuid4())
            base_metadata["file_id"] = document_id

        print(f"Processing document ID: {document_id} with {len(chunks)} chunks")

        # Create a database service instance
        from app.services.database import DatabaseService

        db_service = DatabaseService()

        # Create metadata and save each chunk to database
        vector_ids = []
        for i, chunk_text in enumerate(chunks):
            try:
                # Create chunk metadata
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update(
                    {
                        "chunk_index": i,
                        "filename": filename,
                        "content_type": content_type,
                        "text": chunk_text,
                    }
                )

                # First save chunk to database
                print(f"Saving chunk {i} to database")
                chunk = db_service.create_document_chunk(
                    document_id=document_id,
                    chunk_index=i,
                    text=chunk_text,
                    metadata=chunk_metadata,
                )

                # Generate embedding - using the proper return structure
                print(f"Creating embedding for chunk {i}")
                embedding_result = create_embeddings([chunk_text])

                # Extract the embedding from the result
                if "embeddings" in embedding_result and embedding_result["embeddings"]:
                    embedding = embedding_result["embeddings"][0]

                    # Generate ID and store in vector DB
                    vector_id = str(uuid.uuid4())
                    print(f"Storing vector with ID: {vector_id}")
                    store_vectors([embedding], [chunk_metadata], [vector_id])
                    vector_ids.append(vector_id)

                    # Update the chunk in database with embedding ID
                    db_service.update_chunk_embedding(chunk.id, vector_id)
                    print(f"Chunk {i} processed with vector ID: {vector_id}")
                else:
                    print(f"Warning: No embedding generated for chunk {i}")

            except Exception as chunk_error:
                print(f"Error processing chunk {i}: {str(chunk_error)}")
                import traceback

                traceback.print_exc()
                # Continue processing other chunks

        return {
            "filename": filename,
            "chunks": len(chunks),
            "vector_ids": vector_ids,
            "file_id": document_id,
        }
    except Exception as e:
        print(f"Error in process_document: {str(e)}")
        import traceback

        traceback.print_exc()
        raise
