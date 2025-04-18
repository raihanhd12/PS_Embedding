import argparse
import sys
from pathlib import Path

import requests

# Add the parent directory to the path so we can import the app modules
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import configuration and services
try:
    import app.utils.config as config
    from app.services.database import Base, engine
    from app.services.storage import StorageService
    from app.services.vector_db import init_vector_db
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure you're running this script from the project root.")
    sys.exit(1)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Reset all databases for the Embedding API"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm that you want to delete all data",
    )
    return parser.parse_args()


def reset_postgresql():
    """Reset PostgreSQL database"""
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
    """Reset Qdrant vector database"""
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
        success = init_vector_db()
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
    """Reset MinIO bucket"""
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


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    print("⚠️  WARNING: This will delete all your data! ⚠️")
    print("=============================================")

    if not args.confirm:
        confirm = input("Are you sure you want to proceed? Type 'yes' to confirm: ")
        if confirm.lower() != "yes":
            print("Operation canceled.")
            sys.exit(0)

    print("🔄 Starting database reset...")

    # Reset all services
    all_ok = True
    if not reset_postgresql():
        all_ok = False

    if not reset_qdrant():
        all_ok = False

    if not reset_minio():
        all_ok = False

    if all_ok:
        print("\n✅ All databases have been reset successfully")
        print(
            "\n🔄 You can now run 'python scripts/init_db.py' to verify everything works"
        )
    else:
        print("\n❌ Reset failed. Please check the errors above and try again.")
        sys.exit(1)
