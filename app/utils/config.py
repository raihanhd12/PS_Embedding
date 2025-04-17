from dotenv import dotenv_values

# Load environment variables from .env file
config = dotenv_values(".env")

# Database configuration
DB_URL = config.get("POSTGRES_URL", "")

# Model settings
EMBEDDING_MODEL = config.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Vector database settings
QDRANT_URL = config.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = config.get("COLLECTION_NAME", "documents")
VECTOR_SIZE = config.get("VECTOR_SIZE", "384")

# Elasticsearch settings
ES_URL = config.get("ELASTICSEARCH_URL", "http://localhost:9200")
ES_INDEX = config.get("ELASTICSEARCH_INDEX", "documents")
ES_USERNAME = config.get("ELASTICSEARCH_USERNAME", "")
ES_PASSWORD = config.get("ELASTICSEARCH_PASSWORD", "")

# MinIO configuration
MINIO_ENDPOINT = config.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = config.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = config.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET_NAME = config.get("MINIO_BUCKET_NAME", "documents")
MINIO_SECURE = config.get("MINIO_SECURE", "False").lower() == "true"

# Text processing settings
DEFAULT_CHUNK_SIZE = int(config.get("DEFAULT_CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(config.get("DEFAULT_CHUNK_OVERLAP", "200"))


# Hybrid search settings
ENABLE_HYBRID_SEARCH = config.get("ENABLE_HYBRID_SEARCH", "True").lower() == "true"
VECTOR_WEIGHT = float(config.get("VECTOR_WEIGHT", "0.7"))
KEYWORD_WEIGHT = float(config.get("KEYWORD_WEIGHT", "0.3"))

# API settings
API_PORT = int(config.get("API_PORT", "8001"))
API_HOST = config.get("API_HOST", "0.0.0.0")
API_KEY = config.get("API_KEY", "")


def print_config():
    """Print current configuration settings"""
    print("\n📋 Current Configuration:")
    print(f"  • Database: {DB_URL}")
    print(f"  • Embedding Model: {EMBEDDING_MODEL}")
    print(f"  • Qdrant: {QDRANT_URL}, Collection: {COLLECTION_NAME}")
    print(f"  • Elasticsearch: {ES_URL}, Index: {ES_INDEX}")
    print(f"  • MinIO: {MINIO_ENDPOINT}, Bucket: {MINIO_BUCKET_NAME}")
    print(f"  • API: {API_HOST}:{API_PORT}")
    print(f"  • Hybrid Search: {'Enabled' if ENABLE_HYBRID_SEARCH else 'Disabled'}")
    print(
        f"  • Text Chunking: Size={DEFAULT_CHUNK_SIZE}, Overlap={DEFAULT_CHUNK_OVERLAP}"
    )
