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

# MinIO configuration
MINIO_ENDPOINT = config.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = config.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = config.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET_NAME = config.get("MINIO_BUCKET_NAME", "documents")
MINIO_SECURE = config.get("MINIO_SECURE", "False").lower() == "true"

# Text processing settings
DEFAULT_CHUNK_SIZE = int(config.get("DEFAULT_CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(config.get("DEFAULT_CHUNK_OVERLAP", "200"))

# API settings
API_PORT = int(config.get("API_PORT", "8001"))
API_HOST = config.get("API_HOST", "0.0.0.0")
API_KEY = config.get("API_KEY", "")
