"""
Storage service for document storage using MinIO
"""
from typing import Optional, List, Dict, Any, Tuple
import io
import uuid
from minio import Minio
from minio.error import S3Error
import os
import tempfile

# MinIO configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "documents")
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() == "true"


class StorageService:
    """Storage service for documents using MinIO"""

    def __init__(self):
        """Initialize MinIO client"""
        self.client = Minio(
            endpoint=MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self) -> bool:
        """
        Ensure bucket exists, create if it doesn't

        Returns:
            bool: True if successful
        """
        try:
            if not self.client.bucket_exists(MINIO_BUCKET_NAME):
                self.client.make_bucket(MINIO_BUCKET_NAME)
                print(f"Created bucket: {MINIO_BUCKET_NAME}")
            return True
        except S3Error as e:
            print(f"Error ensuring bucket exists: {e}")
            return False

    async def upload_file(self, file_content: bytes, filename: str, content_type: str,
                          metadata: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
        """
        Upload a file to MinIO

        Args:
            file_content: File content as bytes
            filename: Original filename
            content_type: Content type (MIME)
            metadata: Optional metadata

        Returns:
            Tuple[bool, str]: Success flag and object name
        """
        try:
            # Generate unique object name
            ext = os.path.splitext(filename)[1]
            object_name = f"{uuid.uuid4()}{ext}"

            # Upload to MinIO
            self.client.put_object(
                bucket_name=MINIO_BUCKET_NAME,
                object_name=object_name,
                data=io.BytesIO(file_content),
                length=len(file_content),
                content_type=content_type,
                metadata=metadata
            )

            return True, object_name
        except Exception as e:
            print(f"Error uploading file: {e}")
            return False, ""

    def get_file_url(self, object_name: str, expires: int = 3600) -> Optional[str]:
        """
        Generate a presigned URL for object access

        Args:
            object_name: Name of the object
            expires: Expiration time in seconds

        Returns:
            str: Presigned URL or None if failed
        """
        try:
            url = self.client.presigned_get_object(
                bucket_name=MINIO_BUCKET_NAME,
                object_name=object_name,
                expires=expires
            )
            return url
        except S3Error as e:
            print(f"Error generating presigned URL: {e}")
            return None

    def get_file_content(self, object_name: str) -> Optional[bytes]:
        """
        Get file content as bytes

        Args:
            object_name: Name of the object

        Returns:
            bytes: File content or None if failed
        """
        try:
            response = self.client.get_object(
                bucket_name=MINIO_BUCKET_NAME,
                object_name=object_name
            )
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            print(f"Error getting file content: {e}")
            return None

    def delete_file(self, object_name: str) -> bool:
        """
        Delete a file from MinIO

        Args:
            object_name: Name of the object

        Returns:
            bool: True if successful
        """
        try:
            self.client.remove_object(
                bucket_name=MINIO_BUCKET_NAME,
                object_name=object_name
            )
            return True
        except S3Error as e:
            print(f"Error deleting file: {e}")
            return False

    def list_objects(self) -> List[Dict[str, Any]]:
        """
        List all objects in the bucket

        Returns:
            List[Dict[str, Any]]: List of objects with metadata
        """
        try:
            objects = self.client.list_objects(
                MINIO_BUCKET_NAME, recursive=True)
            result = []

            for obj in objects:
                result.append({
                    "name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified
                })

            return result
        except S3Error as e:
            print(f"Error listing objects: {e}")
            return []

    def get_file_path(self, storage_path):
        """
        Get a local file path for a stored file

        Args:
            storage_path: The storage path of the file

        Returns:
            str: Path to a temporary local file
        """
        # Get the file content
        file_content = self.get_file_content(storage_path)
        if not file_content:
            raise FileNotFoundError(f"File not found: {storage_path}")

        # Create a temporary file with the proper extension
        _, ext = os.path.splitext(storage_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(file_content)
            return temp_file.name
