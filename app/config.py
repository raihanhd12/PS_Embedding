"""
Comprehensive configuration settings for the Embedding API
Uses python-dotenv to load environment variables from .env file
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the absolute path of the current directory
BASE_DIR = Path(__file__).resolve().parent

# Database configuration
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
DB_NAME = os.getenv("POSTGRES_DB", "embeddings")
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Model settings
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Vector database settings
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

# Elasticsearch settings
ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ELASTICSEARCH_INDEX", "documents")
ES_USERNAME = os.getenv("ELASTICSEARCH_USERNAME", "")
ES_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", "")

# MinIO configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "documents")
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() == "true"

# Text processing settings
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))

# Hybrid search settings
ENABLE_HYBRID_SEARCH = os.getenv(
    "ENABLE_HYBRID_SEARCH", "True").lower() == "true"
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "0.3"))

# API settings
API_PORT = int(os.getenv("API_PORT", "8001"))
API_HOST = os.getenv("API_HOST", "0.0.0.0")

def print_config():
    """Print current configuration settings"""
    print("\n📋 Current Configuration:")
    print(f"  • Database: {DB_URL}")
    print(f"  • Embedding Model: {EMBEDDING_MODEL}")
    print(f"  • Qdrant: {QDRANT_URL}, Collection: {COLLECTION_NAME}")
    print(f"  • Elasticsearch: {ES_URL}, Index: {ES_INDEX}")
    print(f"  • MinIO: {MINIO_ENDPOINT}, Bucket: {MINIO_BUCKET_NAME}")
    print(f"  • API: {API_HOST}:{API_PORT}")
    print(
        f"  • Hybrid Search: {'Enabled' if ENABLE_HYBRID_SEARCH else 'Disabled'}")
    print(
        f"  • Text Chunking: Size={DEFAULT_CHUNK_SIZE}, Overlap={DEFAULT_CHUNK_OVERLAP}")
