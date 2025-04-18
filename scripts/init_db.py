import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

# Add the parent directory to the path so we can import the app modules
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import configuration and services
try:
    import app.utils.config as config
    from app.services.database import DatabaseService
    from app.services.storage import StorageService
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure you're running this script from the project root.")
    sys.exit(1)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Initialize all databases for the Embedding API"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreate all databases (warning: this will delete all data)",
    )
    return parser.parse_args()


def ensure_postgres_db():
    """Ensure PostgreSQL database exists locally"""
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

        DB_URL = config.DB_URL

        # Connect to default postgres database first
        conn = psycopg2.connect(dsn=DB_URL)
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
        print(
            f"✅ Qdrant is accessible, collection will be created on application startup"
        )
        return True
    except Exception as e:
        print(f"❌ Error initializing Qdrant: {e}")
        return False


def initialize_minio():
    """Initialize MinIO bucket"""
    print("\n🔄 Initializing MinIO storage...")
    try:
        # Initialize MinIO service
        StorageService()
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
    DB_URL = config.DB_URL

    # Parse DB_URL
    parsed_url = urlparse(DB_URL)

    # Get username and password from parsed URL
    db_user = parsed_url.username
    db_password = parsed_url.password or ""

    """Print connection information for all services"""
    print("\n📋 Connection Information:")
    print(f"  • PostgreSQL: {config.DB_URL}")
    print(f"  • Qdrant: {config.QDRANT_URL}")
    print(f"  • MinIO: {config.MINIO_ENDPOINT}/{config.MINIO_BUCKET_NAME}")
    print("\n🔐 Credentials (from .env):")
    print(f"  • PostgreSQL: {db_user}:{'*' * len(db_password)}")
    print(f"  • MinIO: {config.MINIO_ACCESS_KEY}:{'*' * len(config.MINIO_SECRET_KEY)}")

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
        print("  1. Start the API server with 'python main.py'")
        print("  2. Access the API documentation at http://localhost:8001/docs")
        print("  3. Start embedding your documents!")
    else:
        print(
            "\n⚠️ Some services failed to initialize but you may still be able to proceed"
        )
        print("Try starting the application with:")
        print("  python main.py")
