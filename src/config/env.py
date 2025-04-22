from dotenv import dotenv_values

# Load environment variables from .env file
env = dotenv_values(".env")

# Database configuration
DB_URL = env.get("POSTGRES_URL", "")

# Model settings
EMBEDDING_MODEL = env.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Vector database settings
QDRANT_URL = env.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = env.get("COLLECTION_NAME", "documents")
VECTOR_SIZE = env.get("VECTOR_SIZE", "384")

# MinIO configuration
MINIO_ENDPOINT = env.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = env.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = env.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET_NAME = env.get("MINIO_BUCKET_NAME", "documents")
MINIO_SECURE = env.get("MINIO_SECURE", "False").lower() == "true"

# Text processing settings
DEFAULT_CHUNK_SIZE = int(env.get("DEFAULT_CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(env.get("DEFAULT_CHUNK_OVERLAP", "200"))

# API settings
API_PORT = int(env.get("API_PORT", "8001"))
API_HOST = env.get("API_HOST", "0.0.0.0")
API_KEY = env.get("API_KEY", "")
