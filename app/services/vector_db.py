"""
Vector database service using Qdrant
"""

from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

import app.utils.config as config

# Initialize Qdrant client
client = QdrantClient(url=config.QDRANT_URL)


def init_vector_db() -> bool:
    """
    Initialize vector database collection

    Returns:
        True if successful
    """
    try:
        # Use a fixed vector size instead of importing the embedding model
        # which may not be available due to huggingface_hub issues
        vector_size = (
            config.VECTOR_SIZE
        )  # Common size for sentence-transformers/all-MiniLM-L6-v2

        collections = client.get_collections().collections
        if config.COLLECTION_NAME not in [c.name for c in collections]:
            client.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )
            print(f"Created collection: {config.COLLECTION_NAME}")
        return True
    except Exception as e:
        print(f"Error initializing vector database: {e}")
        return False


def store_vectors(
    vectors: List[List[float]], metadata_list: List[Dict[str, Any]], ids: List[str]
) -> List[str]:
    """
    Store vectors in Qdrant

    Args:
        vectors: List of embedding vectors
        metadata_list: List of metadata dictionaries
        ids: List of vector IDs

    Returns:
        List of vector IDs
    """
    try:
        # Create points
        points = [
            models.PointStruct(id=id, vector=vector, payload=metadata)
            for id, vector, metadata in zip(ids, vectors, metadata_list)
        ]

        # Store in Qdrant
        client.upsert(collection_name=config.COLLECTION_NAME, points=points)

        return ids
    except Exception as e:
        print(f"Error storing vectors: {e}")
        return []


def search_vectors(
    query_vector: List[float],
    limit: int = 5,
    filter_conditions: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Search vectors in Qdrant with filter conditions

    Args:
        query_vector: Query embedding vector
        limit: Maximum number of results
        filter_conditions: Optional filter conditions (e.g., {'active': True})

    Returns:
        List of search results with id, score, and metadata
    """
    try:
        print(f"search_vectors filter_conditions: {filter_conditions}")

        qdrant_filter = None
        if filter_conditions:
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(
                            value=value
                        ),  # FIXED: removed "metadata." prefix
                    )
                    for key, value in filter_conditions.items()
                ]
            )
            print(f"Qdrant filter applied: {qdrant_filter}")

        results = client.search(
            collection_name=config.COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=limit,
        )

        formatted_results = [
            {"id": str(hit.id), "score": hit.score, "metadata": hit.payload or {}}
            for hit in results
        ]
        print(f"search_vectors results: {len(formatted_results)} items")
        for result in formatted_results:
            print(
                f"Result: id={result['id']}, file_id={result['metadata'].get('file_id')}, active={result['metadata'].get('active')}"
            )

        return formatted_results
    except Exception as e:
        print(f"Error in search_vectors: {str(e)}")
        return []


def delete_vector(vector_id: str) -> bool:
    """
    Delete a vector by ID

    Args:
        vector_id: ID of vector to delete

    Returns:
        True if successful
    """
    try:
        client.delete(
            collection_name=config.COLLECTION_NAME,
            points_selector=models.PointIdsList(points=[vector_id]),
        )
        return True
    except Exception as e:
        print(f"Error deleting vector: {e}")
        return False


def delete_vectors_by_filter(filter_conditions: Dict[str, Any]) -> bool:
    """
    Delete vectors matching filter conditions

    Args:
        filter_conditions: Filter conditions

    Returns:
        True if successful
    """
    try:
        # Build filter
        filter_query = models.Filter(
            must=[
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(
                        value=value
                    ),  # FIXED: removed "metadata." prefix
                )
                for key, value in filter_conditions.items()
            ]
        )

        # Delete from Qdrant
        client.delete(
            collection_name=config.COLLECTION_NAME,
            points_selector=models.FilterSelector(filter=filter_query),
        )
        return True
    except Exception as e:
        print(f"Error deleting vectors: {e}")
        return False


def update_vectors_metadata(
    filter_conditions: Dict[str, Any], metadata_update: Dict[str, Any]
) -> bool:
    """
    Update metadata for vectors matching filter conditions

    Args:
        filter_conditions: Filter to match vectors
        metadata_update: Metadata fields to update

    Returns:
        bool: True if successful
    """
    try:
        # Build filter - FIXED: removed "metadata." prefix
        must_conditions = [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in filter_conditions.items()
        ]
        filter_query = models.Filter(must=must_conditions)

        # Use set_payload with a FilterSelector
        client.set_payload(
            collection_name=config.COLLECTION_NAME,
            payload=metadata_update,
            points=models.FilterSelector(filter=filter_query),
        )
        print(
            f"Updated Qdrant metadata for filter: {filter_conditions}, update: {metadata_update}"
        )
        return True
    except Exception as e:
        print(f"Error updating vector metadata: {e}")
        return False
