"""
Text embedding service using sentence-transformers
"""
from typing import List, Dict, Any, Optional
import uuid
from sentence_transformers import SentenceTransformer

import app.utils.config as config
from app.services.vector_db import store_vectors

# Initialize embedding model
model = SentenceTransformer(config.EMBEDDING_MODEL)
embedding_dimension = model.get_sentence_embedding_dimension()


def get_dimension() -> int:
    """Get the embedding dimension of the model"""
    return embedding_dimension


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for list of texts

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors
    """
    if not texts:
        return []

    # Generate embeddings
    embeddings = model.encode(texts, convert_to_numpy=True).tolist()
    return embeddings


def create_embeddings(
    texts: List[str],
    metadata: Optional[List[Dict[str, Any]]] = None,
    store: bool = False
) -> Dict[str, Any]:
    """
    Embed texts and optionally store in vector database

    Args:
        texts: List of text strings to embed
        metadata: Optional metadata for each text
        store: Whether to store embeddings in vector DB

    Returns:
        Dictionary with embedding information and optional vector IDs
    """
    # Generate embeddings
    embeddings = embed_texts(texts)

    result = {
        "count": len(embeddings),
        "dimension": embedding_dimension,
    }

    # Store embeddings if requested
    if store:
        # Ensure we have metadata for each text
        if not metadata or len(metadata) != len(texts):
            metadata_list = [
                {"text": text, "id": str(uuid.uuid4())} for text in texts]
        else:
            metadata_list = metadata
            # Ensure each metadata item has the text
            for i, md in enumerate(metadata_list):
                if "text" not in md:
                    md["text"] = texts[i]

        # Generate IDs if not provided
        ids = [md.get("id", str(uuid.uuid4())) for md in metadata_list]

        # Store in vector DB
        store_vectors(embeddings, metadata_list, ids)

        # Return IDs
        result["vector_ids"] = ids
    else:
        # Include embeddings only if not storing them
        result["embeddings"] = embeddings

    return result
