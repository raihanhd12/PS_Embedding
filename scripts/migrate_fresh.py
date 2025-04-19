# scripts/migrate_fresh.py
import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

current_dir = Path(__file__).resolve().parent  # scripts/
project_root = current_dir.parent  # PS_Embedding/
sys.path.insert(0, str(project_root.resolve()))

try:
    import app.utils.config as config
    from app.services.database import Base, DatabaseService, engine
    from app.services.storage import StorageService
    from app.services.vector_db import VectorDatabaseService
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure you are running this script from the project root directory.")
    sys.exit(1)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Drop and reinitialize all databases and storage for the Embedding API "
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm that you want to delete all data and reinitialize",
    )
    return parser.parse_args()


def reset_postgresql():
    """Reset PostgreSQL database by dropping and recreating tables"""
    print("\n🔄 Resetting PostgreSQL database...")
    try:
        # Drop all tables
        print("🗑️ Dropping all tables...")
        Base.metadata.drop_all(bind=engine)
        print("✅ All tables dropped")

        # Recreate tables
        print("🔄 Recreating tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables recreated")
        return True
    except Exception as e:
        print(f"❌ Error resetting PostgreSQL: {e}")
        return False


def reset_qdrant():
    """Reset Qdrant vector database by deleting and recreating the collection"""
    print("\n🔄 Resetting Qdrant vector database...")
    try:
        # Delete collection if it exists
        print(f"🗑️ Deleting collection '{config.COLLECTION_NAME}'...")
        response = requests.delete(
            f"{config.QDRANT_URL}/collections/{config.COLLECTION_NAME}"
        )

        if response.status_code in (200, 404):
            print(f"✅ Collection '{config.COLLECTION_NAME}' deleted or not found")
        else:
            print(f"❌ Failed to delete collection: {response.status_code}")
            print(response.text)
            return False

        # Recreate collection
        print(f"🔄 Recreating collection '{config.COLLECTION_NAME}'...")
        vector_db_service = VectorDatabaseService()
        success = vector_db_service.init_vector_db()
        if success:
            print(f"✅ Collection '{config.COLLECTION_NAME}' recreated")
            return True
        else:
            print("❌ Failed to recreate collection")
            return False
    except Exception as e:
        print(f"❌ Error resetting Qdrant: {e}")
        return False


def reset_minio():
    """Reset MinIO bucket by deleting all objects and ensuring the bucket exists"""
    print("\n🔄 Resetting MinIO storage...")
    try:
        # Create storage service
        storage = StorageService()

        # Delete all objects
        print(f"🗑️ Deleting all objects in bucket '{config.MINIO_BUCKET_NAME}'...")
        objects = storage.list_objects()
        if objects:
            for obj in objects:
                storage.delete_file(obj["name"])
                print(f"  • Deleted object: {obj['name']}")
            print(f"✅ All {len(objects)} objects deleted")
        else:
            print("✅ No objects to delete")

        # Ensure bucket exists
        storage._ensure_bucket_exists()
        print(f"✅ MinIO bucket '{config.MINIO_BUCKET_NAME}' reset")
        return True
    except Exception as e:
        print(f"❌ Error resetting MinIO: {e}")
        return False


def ensure_postgres_db():
    """Ensure PostgreSQL database exists locally"""
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

        # Connect to default postgres database
        conn = psycopg2.connect(dsn=config.DB_URL)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        # Create a cursor
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", ("embeddings",))
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
    """Initialize PostgreSQL database by ensuring it exists and resetting tables"""
    print("\n🔄 Initializing PostgreSQL database...")

    # First ensure database exists
    if not ensure_postgres_db():
        print("⚠️ Cannot proceed with PostgreSQL initialization without database")
        return False

    # Reset tables (drop and recreate)
    return reset_postgresql()


def initialize_qdrant():
    """Initialize Qdrant vector database by resetting the collection"""
    return reset_qdrant()


def initialize_minio():
    """Initialize MinIO bucket by resetting it"""
    return reset_minio()


def verify_all_services():
    """Verify that all services are running"""
    print("\n🔍 Verifying all services...")
    all_ok = True

    # Check PostgreSQL
    try:
        DatabaseService.get_documents(limit=1)
        print("✅ PostgreSQL database is operational")
    except Exception as e:
        print(f"❌ PostgreSQL check failed: {e}")
        all_ok = False

    # Check Qdrant
    try:
        response = requests.get(f"{config.QDRANT_URL}/collections")
        if response.status_code == 200:
            collections = response.json().get("result", {}).get("collections", [])
            print(f"✅ Qdrant is operational with {len(collections)} collections")
        else:
            print(f"❌ Qdrant check failed, status code: {response.status_code}")
            all_ok = False
    except requests.RequestException as e:
        print(f"❌ Qdrant check failed: {e}")
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
    parsed_url = urlparse(config.DB_URL)
    db_user = parsed_url.username

    print("\n📋 Connection Information:")
    print(f"  • PostgreSQL: {config.DB_URL}")
    print(f"  • Qdrant: {config.QDRANT_URL}")
    print(f"  • MinIO: {config.MINIO_ENDPOINT}/{config.MINIO_BUCKET_NAME}")
    print("\n🔐 Credentials (from .env):")
    print(f"  • PostgreSQL: {db_user}:*****")
    print(f"  • MinIO: {config.MINIO_ACCESS_KEY}:*****")

    print("\n🌐 Admin Interfaces:")
    print(f"  • Qdrant Dashboard: {config.QDRANT_URL.rstrip('/')}/dashboard")

    # Extract host and port from MINIO_ENDPOINT
    minio_host = config.MINIO_ENDPOINT.split(":")[0]
    minio_console_url = f"http://{minio_host}:9001"
    print(f"  • MinIO Console: {minio_console_url}")
    print(f"     - Username: {config.MINIO_ACCESS_KEY}")
    print(f"     - Password: {config.MINIO_SECRET_KEY}")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    print("⚠️  WARNING: This will delete all data and reinitialize all services! ⚠️")
    print("================================================================")

    if not args.confirm:
        confirm = input("Are you sure you want to proceed? Type 'yes' to confirm: ")
        if confirm.lower() != "yes":
            print("Operation canceled.")
            sys.exit(0)

    print("🔄 Starting fresh migration...")

    # Initialize (reset and recreate) all services
    all_ok = True
    if not initialize_postgresql():
        all_ok = False

    if not initialize_qdrant():
        all_ok = False

    if not initialize_minio():
        all_ok = False

    if all_ok:
        print("\n✅ All services have been reset and reinitialized successfully")
        # Verify all services
        if verify_all_services():
            print("\n🎉 All services are operational")
        else:
            print("\n⚠️ One or more services have issues, check the output above")

        # Print connection info
        print_connection_info()

        print("\n📝 Next steps:")
        print("  1. Start the API server with 'python main.py'")
        print(
            f"  2. Access the API documentation at http://localhost:{config.API_PORT}/docs"
        )
        print("  3. Start embedding your documents!")
    else:
        print("\n❌ Migration failed. Please check the errors above and try again.")
        sys.exit(1)
