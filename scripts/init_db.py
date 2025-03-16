#!/usr/bin/env python
"""
Database initialization script for the Embedding API

This script initializes all required databases and services:
- PostgreSQL database and tables
- Qdrant vector database collections
- Elasticsearch indices
- MinIO buckets

Usage:
    python scripts/init_db.py [--force]

Options:
    --force    Force recreate all databases (warning: this will delete all data)
"""
import os
import sys
import time
import argparse
from pathlib import Path
import requests

# Add the parent directory to the path so we can import the app modules
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import configuration and services
try:
    import config
    from services.database import DatabaseService
    from services.storage import StorageService
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure you're running this script from the project root.")
    sys.exit(1)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Initialize all databases for the Embedding API")
    parser.add_argument("--force", action="store_true",
                        help="Force recreate all databases (warning: this will delete all data)")
    return parser.parse_args()


def ensure_postgres_db():
    """Ensure PostgreSQL database exists locally"""
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

        # Connect to default postgres database first
        conn = psycopg2.connect(
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            host=config.DB_HOST,
            port=config.DB_PORT,
            database="postgres"  # Connect to default DB first
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        # Create a cursor
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", ("embeddings",))
        exists = cursor.fetchone()

        if not exists:
            print("🔄 Creating PostgreSQL database 'embeddings'...")
            cursor.execute("CREATE DATABASE embeddings")
            print("✅ Database 'embeddings' created successfully")
        else:
            print("✅ PostgreSQL database 'embeddings' already exists")

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error creating PostgreSQL database: {e}")
        return False


def initialize_postgresql():
    """Initialize PostgreSQL database"""
    print("\n🔄 Initializing PostgreSQL database...")

    # First ensure database exists
    if not ensure_postgres_db():
        print("⚠️ Cannot proceed with PostgreSQL initialization without database")
        return False

    try:
        # Initialize database tables
        DatabaseService.init_db()
        print("✅ PostgreSQL database tables initialized")
        return True
    except Exception as e:
        print(f"❌ Error initializing PostgreSQL: {e}")
        return False


def initialize_qdrant():
    """Initialize Qdrant vector database"""
    print("\n🔄 Initializing Qdrant vector database...")
    try:
        # Check if Qdrant is running
        try:
            response = requests.get(f"{config.QDRANT_URL}/collections")
            if response.status_code != 200:
                print(f"❌ Qdrant returned status code {response.status_code}")
                return False
        except requests.RequestException as e:
            print(f"❌ Cannot connect to Qdrant at {config.QDRANT_URL}: {e}")
            print("Make sure Qdrant is running and accessible.")
            return False

        # Initialize vector database - we'll skip this step temporarily
        # and let the application create the collection on startup
        print(f"✅ Qdrant is accessible, collection will be created on application startup")
        return True
    except Exception as e:
        print(f"❌ Error initializing Qdrant: {e}")
        return False


def initialize_elasticsearch(args):
    """Initialize Elasticsearch indices"""
    print("\n🔄 Initializing Elasticsearch...")
    try:
        # Check if Elasticsearch is running
        try:
            auth = None
            if config.ES_USERNAME and config.ES_PASSWORD:
                auth = (config.ES_USERNAME, config.ES_PASSWORD)

            response = requests.get(f"{config.ES_URL}", auth=auth)
            if response.status_code != 200:
                print(
                    f"❌ Elasticsearch returned status code {response.status_code}")
                return False
        except requests.RequestException as e:
            print(f"❌ Cannot connect to Elasticsearch at {config.ES_URL}: {e}")
            print("Make sure Elasticsearch is running and accessible.")
            return False

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

        # Check if index exists and delete if force option is specified
        index_exists = requests.head(
            f"{config.ES_URL}/{config.ES_INDEX}",
            auth=auth
        ).status_code == 200

        if index_exists and args:
            print(
                f"🗑️ Deleting existing Elasticsearch index '{config.ES_INDEX}'...")
            delete_response = requests.delete(
                f"{config.ES_URL}/{config.ES_INDEX}",
                auth=auth
            )
            if delete_response.status_code not in (200, 404):
                print(
                    f"❌ Failed to delete Elasticsearch index: {delete_response.status_code}")
                print(delete_response.text)
                return False
            index_exists = False

        if not index_exists:
            create_response = requests.put(
                f"{config.ES_URL}/{config.ES_INDEX}",
                json=mapping,
                auth=auth
            )

            if create_response.status_code not in (200, 201):
                print(
                    f"❌ Failed to create Elasticsearch index: {create_response.status_code}")
                print(create_response.text)
                return False
            print(f"✅ Elasticsearch index '{config.ES_INDEX}' created")
        else:
            print(f"✅ Elasticsearch index '{config.ES_INDEX}' already exists")

        return True
    except Exception as e:
        print(f"❌ Error initializing Elasticsearch: {e}")
        return False


def initialize_minio():
    """Initialize MinIO bucket"""
    print("\n🔄 Initializing MinIO storage...")
    try:
        # Initialize MinIO service
        storage_service = StorageService()
        print(f"✅ MinIO bucket '{config.MINIO_BUCKET_NAME}' initialized")
        return True
    except Exception as e:
        print(f"❌ Error initializing MinIO: {e}")
        return False


def verify_all_services():
    """Verify that all services are running"""
    print("\n🔍 Verifying all services...")
    all_ok = True

    # Check PostgreSQL
    try:
        result = DatabaseService.get_documents(limit=1)
        print("✅ PostgreSQL database is operational")
    except Exception as e:
        print(f"❌ PostgreSQL check failed: {e}")
        all_ok = False

    # Check Qdrant
    try:
        response = requests.get(f"{config.QDRANT_URL}/collections")
        if response.status_code == 200:
            collections = response.json().get("result", {}).get("collections", [])
            print(
                f"✅ Qdrant is operational with {len(collections)} collections")
        else:
            print(
                f"❌ Qdrant check failed, status code: {response.status_code}")
            all_ok = False
    except requests.RequestException as e:
        print(f"❌ Qdrant check failed: {e}")
        all_ok = False

    # Check Elasticsearch
    try:
        auth = None
        if config.ES_USERNAME and config.ES_PASSWORD:
            auth = (config.ES_USERNAME, config.ES_PASSWORD)

        response = requests.get(f"{config.ES_URL}/_cluster/health", auth=auth)
        if response.status_code == 200:
            status = response.json().get("status")
            print(f"✅ Elasticsearch is operational with status '{status}'")
        else:
            print(
                f"❌ Elasticsearch check failed, status code: {response.status_code}")
            all_ok = False
    except requests.RequestException as e:
        print(f"❌ Elasticsearch check failed: {e}")
        all_ok = False

    # Check MinIO
    try:
        storage = StorageService()
        objects = storage.list_objects()
        print(f"✅ MinIO is operational with {len(objects)} objects")
    except Exception as e:
        print(f"❌ MinIO check failed: {e}")
        all_ok = False

    return all_ok


def print_connection_info():
    """Print connection information for all services"""
    print("\n📋 Connection Information:")
    print(f"  • PostgreSQL: {config.DB_URL}")
    print(f"  • Qdrant: {config.QDRANT_URL}")
    print(f"  • Elasticsearch: {config.ES_URL}/{config.ES_INDEX}")
    print(f"  • MinIO: {config.MINIO_ENDPOINT}/{config.MINIO_BUCKET_NAME}")
    print("\n🔐 Credentials (from .env):")
    print(f"  • PostgreSQL: {config.DB_USER}:{'*' * len(config.DB_PASSWORD)}")
    print(
        f"  • MinIO: {config.MINIO_ACCESS_KEY}:{'*' * len(config.MINIO_SECRET_KEY)}")
    if config.ES_USERNAME:
        print(
            f"  • Elasticsearch: {config.ES_USERNAME}:{'*' * len(config.ES_PASSWORD)}")

    print("\n🌐 Admin Interfaces:")
    print(f"  • Qdrant Dashboard: {config.QDRANT_URL.rstrip('/')}/dashboard")
    # Extract host and port from ES_URL
    es_host = config.ES_URL.replace(
        "http://", "").replace("https://", "").split("/")[0]
    kibana_url = f"http://{es_host.split(':')[0]}:5601"
    print(f"  • Kibana (Elasticsearch): {kibana_url}")

    # Extract host and port from MINIO_ENDPOINT
    minio_host = config.MINIO_ENDPOINT.split(":")[0]
    minio_console_url = f"http://{minio_host}:9001"
    print(f"  • MinIO Console: {minio_console_url}")
    print(f"     - Username: {config.MINIO_ACCESS_KEY}")
    print(f"     - Password: {config.MINIO_SECRET_KEY}")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    print("🚀 Embedding API Database Initialization")
    print("========================================")

    # Print current config
    config.print_config()

    # Initialize all services
    all_ok = True
    if not initialize_postgresql():
        all_ok = False

    if not initialize_qdrant():
        all_ok = False

    if not initialize_elasticsearch(True):
        all_ok = False

    if not initialize_minio():
        all_ok = False

    if all_ok:
        print("\n✅ All services initialized successfully")
        # Verify all services
        if verify_all_services():
            print("\n🎉 All services are operational")
        else:
            print("\n⚠️ One or more services have issues, check the output above")

        # Print connection info
        print_connection_info()

        print("\n📝 Next steps:")
        print("  1. Start the API server with 'uvicorn app:app --reload'")
        print("  2. Access the API documentation at http://localhost:8000/docs")
        print("  3. Start embedding your documents!")
    else:
        print(
            "\n⚠️ Some services failed to initialize but you may still be able to proceed")
        print("Try starting the application with:")
        print("  uvicorn app:app --reload")
