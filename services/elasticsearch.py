"""
Elasticsearch service for keyword search and hybrid search
"""
from typing import List, Dict, Any, Optional
import requests
import json
import os

# Elasticsearch configuration
ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ELASTICSEARCH_INDEX", "documents")
ES_USERNAME = os.getenv("ELASTICSEARCH_USERNAME", "")
ES_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", "")


class ElasticsearchService:
    """Service for Elasticsearch operations"""

    def __init__(self):
        """Initialize Elasticsearch service"""
        self.base_url = ES_URL
        self.index = ES_INDEX
        self.auth = None

        # Set up authentication if provided
        if ES_USERNAME and ES_PASSWORD:
            self.auth = (ES_USERNAME, ES_PASSWORD)

        # Initialize index
        self._init_index()

    def _init_index(self) -> bool:
        """
        Initialize index with proper mappings

        Returns:
            bool: True if successful
        """
        try:
            # Check if index exists
            response = requests.head(
                f"{self.base_url}/{self.index}",
                auth=self.auth
            )

            if response.status_code != 200:
                # Create index with mappings
                mapping = {
                    "mappings": {
                        "properties": {
                            "file_id": {"type": "keyword"},
                            "chunk_index": {"type": "integer"},
                            "filename": {"type": "keyword"},
                            "content_type": {"type": "keyword"},
                            "text": {
                                "type": "text",
                                "analyzer": "standard"
                            },
                            "embedding_id": {"type": "keyword"},
                            "metadata": {"type": "object"}
                        }
                    },
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    }
                }

                create_response = requests.put(
                    f"{self.base_url}/{self.index}",
                    json=mapping,
                    auth=self.auth
                )

                if create_response.status_code in (200, 201):
                    print(f"Created Elasticsearch index: {self.index}")
                    return True
                else:
                    print(
                        f"Failed to create Elasticsearch index: {create_response.status_code}")
                    print(create_response.text)
                    return False

            return True
        except Exception as e:
            print(f"Error initializing Elasticsearch index: {e}")
            return False

    def index_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """
        Index a document in Elasticsearch

        Args:
            doc_id: Document ID
            document: Document data

        Returns:
            bool: True if successful
        """
        try:
            response = requests.put(
                f"{self.base_url}/{self.index}/_doc/{doc_id}",
                json=document,
                auth=self.auth
            )

            return response.status_code in (200, 201)
        except Exception as e:
            print(f"Error indexing document: {e}")
            return False

    def bulk_index(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Bulk index multiple documents

        Args:
            documents: List of documents with IDs

        Returns:
            bool: True if successful
        """
        try:
            bulk_data = []

            for doc in documents:
                # Each document should have an 'id' field
                doc_id = doc.pop('id', None)
                if not doc_id:
                    continue

                # Add index action and document
                bulk_data.append(
                    {"index": {"_index": self.index, "_id": doc_id}})
                bulk_data.append(doc)

            if not bulk_data:
                return False

            # Convert to newline-delimited JSON
            bulk_body = "\n".join([json.dumps(item)
                                  for item in bulk_data]) + "\n"

            # Send bulk request
            response = requests.post(
                f"{self.base_url}/_bulk",
                headers={"Content-Type": "application/x-ndjson"},
                data=bulk_body,
                auth=self.auth
            )

            result = response.json()
            if result.get("errors", True):
                print(f"Bulk indexing had errors: {result}")
                return False

            return True
        except Exception as e:
            print(f"Error bulk indexing: {e}")
            return False

    def search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search documents by keyword

        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional filters

        Returns:
            List[Dict[str, Any]]: Search results
        """
        try:
            # Build query
            es_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"text": query}}
                        ]
                    }
                },
                "size": limit
            }

            # Add filters if provided
            if filters:
                filter_list = []
                for key, value in filters.items():
                    filter_list.append({"term": {key: value}})

                if filter_list:
                    es_query["query"]["bool"]["filter"] = filter_list

            # Send search request
            response = requests.post(
                f"{self.base_url}/{self.index}/_search",
                json=es_query,
                auth=self.auth
            )

            if response.status_code != 200:
                print(f"Search failed: {response.status_code}")
                return []

            # Parse results
            result = response.json()
            hits = result.get("hits", {}).get("hits", [])

            # Format results
            return [
                {
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "source": hit["_source"]
                }
                for hit in hits
            ]
        except Exception as e:
            print(f"Error searching Elasticsearch: {e}")
            return []

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID

        Args:
            doc_id: Document ID

        Returns:
            bool: True if successful
        """
        try:
            response = requests.delete(
                f"{self.base_url}/{self.index}/_doc/{doc_id}",
                auth=self.auth
            )

            return response.status_code in (200, 204)
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False

    def delete_by_query(self, query: Dict[str, Any]) -> bool:
        """
        Delete documents matching a query

        Args:
            query: Query object

        Returns:
            bool: True if successful
        """
        try:
            response = requests.post(
                f"{self.base_url}/{self.index}/_delete_by_query",
                json=query,
                auth=self.auth
            )

            return response.status_code == 200
        except Exception as e:
            print(f"Error deleting by query: {e}")
            return False

    def hybrid_search(self, query: str, vector_results: List[Dict[str, Any]],
                      vector_weight: float = 0.7, keyword_weight: float = 0.3,
                      limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and keyword results

        Args:
            query: Search query text
            vector_results: Results from vector search
            vector_weight: Weight for vector search results (0-1)
            keyword_weight: Weight for keyword search results (0-1)
            limit: Maximum number of results to return

        Returns:
            List[Dict[str, Any]]: Combined and reranked results
        """
        try:
            # Get keyword search results
            keyword_results = self.search(query, limit=limit*2)

            # Combine and rerank results
            combined_results = {}

            # Add vector search results with weight
            for result in vector_results:
                doc_id = result["id"]
                combined_results[doc_id] = {
                    "id": doc_id,
                    "text": result.get("metadata", {}).get("text", ""),
                    "score": result["score"] * vector_weight,
                    "metadata": result.get("metadata", {}),
                    "source": "vector"
                }

            # Add keyword search results with weight
            for hit in keyword_results:
                doc_id = hit["id"]
                if doc_id in combined_results:
                    # Document already in results from vector search
                    combined_results[doc_id]["score"] += hit["score"] * \
                        keyword_weight
                    combined_results[doc_id]["source"] = "hybrid"
                else:
                    combined_results[doc_id] = {
                        "id": doc_id,
                        "text": hit["source"].get("text", ""),
                        "score": hit["score"] * keyword_weight,
                        "metadata": {k: v for k, v in hit["source"].items() if k != "text"},
                        "source": "keyword"
                    }

            # Sort by score and limit results
            final_results = sorted(
                combined_results.values(),
                key=lambda x: x["score"],
                reverse=True
            )[:limit]

            return final_results
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            # Fall back to vector search only
            return vector_results[:limit]
