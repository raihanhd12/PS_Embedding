from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

import src.config.env as env


class VectorDatabaseService:
    def __init__(self):
        """
        Inisialisasi koneksi ke Qdrant dan koleksi
        """
        # Initialize Qdrant client
        self.client = QdrantClient(url=env.QDRANT_URL)
        self.collection_name = env.COLLECTION_NAME
        self.vector_size = env.VECTOR_SIZE  # Ukuran vektor umum untuk model embedding

    def init_vector_db(self) -> bool:
        """
        Initialize vector database collection

        Returns:
            True if successful
        """
        try:
            collections = self.client.get_collections().collections
            if self.collection_name not in [c.name for c in collections]:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size, distance=models.Distance.COSINE
                    ),
                )
                print(f"Created collection: {self.collection_name}")
            return True
        except Exception as e:
            print(f"Error initializing vector database: {e}")
            return False

    def store_vectors(
        self,
        vectors: List[List[float]],
        metadata_list: List[Dict[str, Any]],
        ids: List[str],
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
            self.client.upsert(collection_name=self.collection_name, points=points)
            return ids
        except Exception as e:
            print(f"Error storing vectors: {e}")
            return []

    def delete_vector(self, vector_id: str) -> bool:
        """
        Delete a vector by ID

        Args:
            vector_id: ID of vector to delete

        Returns:
            True if successful
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[vector_id]),
            )
            return True
        except Exception as e:
            print(f"Error deleting vector: {e}")
            return False

    def delete_vectors_by_filter(self, filter_conditions: Dict[str, Any]) -> bool:
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
                        match=models.MatchValue(value=value),
                    )
                    for key, value in filter_conditions.items()
                ]
            )

            # Delete from Qdrant
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=filter_query),
            )
            return True
        except Exception as e:
            print(f"Error deleting vectors: {e}")
            return False

    def update_vectors_metadata(
        self, filter_conditions: Dict[str, Any], metadata_update: Dict[str, Any]
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
            # Build filter
            must_conditions = [
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
                for key, value in filter_conditions.items()
            ]
            filter_query = models.Filter(must=must_conditions)

            # Use set_payload with a FilterSelector
            self.client.set_payload(
                collection_name=self.collection_name,
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

    def search_vectors(
        self,
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
                # Build filter - create a list to store conditions
                must_conditions = []

                for key, value in filter_conditions.items():
                    # Make sure to handle tuples correctly - extract the first value if it's a tuple
                    if isinstance(value, tuple) and len(value) > 0:
                        actual_value = value[0]
                    else:
                        actual_value = value

                    # Add the condition with the corrected value
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=actual_value),
                        )
                    )

                # Create the filter with the list of conditions
                qdrant_filter = models.Filter(must=must_conditions)
                print(f"Qdrant filter applied: {qdrant_filter}")

            results = self.client.search(
                collection_name=self.collection_name,
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
