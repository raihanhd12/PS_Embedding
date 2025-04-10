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
        vector_size = 384  # Common size for sentence-transformers/all-MiniLM-L6-v2

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
    Search for similar vectors

    Args:
        query_vector: Query embedding vector
        limit: Maximum number of results
        filter_conditions: Optional filter conditions that can include multiple values

    Returns:
        List of search results
    """
    try:
        # Build filter if provided
        filter_query = None
        if filter_conditions:
            filter_conditions_list = []
            for key, value in filter_conditions.items():
                # Handle multiple values for a single field
                if isinstance(value, list):
                    # For arrays, create a should clause (logical OR)
                    should_conditions = []
                    for val in value:
                        should_conditions.append(
                            models.FieldCondition(
                                key=key, match=models.MatchValue(value=val)
                            )
                        )
                    filter_conditions_list.append(
                        models.Filter(should=should_conditions)
                    )
                else:
                    # For single values, create a regular must clause
                    filter_conditions_list.append(
                        models.FieldCondition(
                            key=key, match=models.MatchValue(value=value)
                        )
                    )

            # Create the final filter with all conditions
            filter_query = models.Filter(must=filter_conditions_list)

        # Search in Qdrant
        search_results = client.search(
            collection_name=config.COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_query,
        )

        # Format results
        return [
            {
                "id": str(result.id),
                "score": float(result.score),
                "metadata": result.payload,
            }
            for result in search_results
        ]
    except Exception as e:
        print(f"Error searching vectors: {e}")
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
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
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
