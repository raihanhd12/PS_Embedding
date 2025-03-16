"""
API endpoints for the Embedding API
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Dict, Any, Optional
import json

from models.schemas import (
    TextEmbeddingRequest,
    SearchRequest,
    SearchResponse,
    EmbeddingResponse,
    DocumentProcessResponse,
    ChunkSearchResult
)
from services.embedding import create_embeddings, embed_texts
from services.vector_db import search_vectors, delete_vector, delete_vectors_by_filter
from services.text_processor import process_document

# Create API router
router = APIRouter(prefix="/api")


@router.post("/embed", response_model=EmbeddingResponse)
async def embed_texts_endpoint(request: TextEmbeddingRequest):
    """
    Generate embeddings for text and optionally store in vector DB
    """
    try:
        result = create_embeddings(
            texts=request.texts,
            metadata=request.metadata,
            store=request.store
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error embedding texts: {str(e)}")


@router.post("/upload-embed", response_model=DocumentProcessResponse)
async def upload_and_embed_endpoint(
    file: UploadFile = File(...),
    chunk_size: int = Form(None),
    chunk_overlap: int = Form(None),
    metadata: Optional[str] = Form(None)
):
    """
    Extract text from document, chunk it, embed chunks, and store in vector DB
    """
    try:
        # Parse metadata if provided
        base_metadata = {}
        if metadata:
            try:
                base_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400, detail="Invalid metadata JSON")

        # Get chunk size and overlap
        from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
        chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
        chunk_overlap = chunk_overlap or DEFAULT_CHUNK_OVERLAP

        # Read file content
        file_content = await file.read()

        # Process document
        result = await process_document(
            file_content=file_content,
            filename=file.filename,
            content_type=file.content_type or "application/octet-stream",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            base_metadata=base_metadata
        )

        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing file: {str(e)}")


@router.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    """
    Search for similar text based on semantic similarity
    """
    try:
        # Generate embedding for query
        query_embedding = embed_texts([request.query])[0]

        # Search vectors
        search_results = search_vectors(
            query_vector=query_embedding,
            limit=request.limit,
            filter_conditions=request.filter_metadata
        )

        # Format results
        results = []
        for result in search_results:
            # Extract text from metadata
            text = result["metadata"].get("text", "")
            # Remove text from metadata to avoid duplication
            metadata = {k: v for k,
                        v in result["metadata"].items() if k != "text"}

            results.append(ChunkSearchResult(
                id=result["id"],
                text=text,
                score=result["score"],
                metadata=metadata
            ))

        return SearchResponse(results=results)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error searching: {str(e)}")


@router.delete("/vectors/{vector_id}")
async def delete_vector_endpoint(vector_id: str):
    """
    Delete a specific vector by ID
    """
    try:
        success = delete_vector(vector_id)
        if success:
            return {"message": f"Vector {vector_id} deleted successfully"}
        else:
            raise HTTPException(
                status_code=500, detail="Failed to delete vector")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting vector: {str(e)}")


@router.delete("/documents/{file_id}")
async def delete_document_endpoint(file_id: str):
    """
    Delete all vectors associated with a file ID
    """
    try:
        success = delete_vectors_by_filter({"file_id": file_id})
        if success:
            return {"message": f"All vectors for file {file_id} deleted successfully"}
        else:
            raise HTTPException(
                status_code=500, detail="Failed to delete vectors")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting vectors: {str(e)}")
