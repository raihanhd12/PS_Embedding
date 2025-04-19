"""
Storage service for document storage using MinIO with organized folders
"""

import io
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple

from minio import Minio
from minio.error import S3Error

import app.utils.config as config

# MinIO configuration
MINIO_ENDPOINT = config.MINIO_ENDPOINT
MINIO_ACCESS_KEY = config.MINIO_ACCESS_KEY
MINIO_SECRET_KEY = config.MINIO_SECRET_KEY
MINIO_BUCKET_NAME = config.MINIO_BUCKET_NAME
MINIO_SECURE = config.MINIO_SECURE


class StorageService:
    """Storage service for documents using MinIO with organized folder structure"""

    # Define folder paths for different content types
    FOLDER_DOCUMENTS = MINIO_BUCKET_NAME + "/"
    FOLDER_IMAGES = MINIO_BUCKET_NAME + "images/"

    # Define specific document type folders
    FOLDER_PDF = MINIO_BUCKET_NAME + "/pdf/"
    FOLDER_DOCX = MINIO_BUCKET_NAME + "/docx/"
    FOLDER_TEXT = MINIO_BUCKET_NAME + "/text/"
    FOLDER_OTHER = MINIO_BUCKET_NAME + "/other/"

    def __init__(self):
        """Initialize MinIO client"""
        self.client = Minio(
            endpoint=MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
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

    def _get_folder_path(
        self, content_type: str, is_extracted_image: bool = False
    ) -> str:
        """
        Determine the appropriate folder path based on content type

        Args:
            content_type: Content type (MIME)
            is_extracted_image: Whether this is an image extracted from a document

        Returns:
            str: Folder path
        """
        if is_extracted_image:
            return self.FOLDER_IMAGES

        content_type_lower = content_type.lower()

        if "pdf" in content_type_lower:
            return self.FOLDER_PDF
        elif "docx" in content_type_lower or "document" in content_type_lower:
            return self.FOLDER_DOCX
        elif "text" in content_type_lower:
            return self.FOLDER_TEXT
        elif "image" in content_type_lower:
            return self.FOLDER_IMAGES
        else:
            return self.FOLDER_OTHER

    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        metadata: Optional[Dict[str, str]] = None,
        is_extracted_image: bool = False,
        document_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Upload a file to MinIO with organized folder structure

        Args:
            file_content: File content as bytes
            filename: Original filename
            content_type: Content type (MIME)
            metadata: Optional metadata
            is_extracted_image: Whether this is an image extracted from a document
            document_id: Optional document ID (used for organizing extracted content)

        Returns:
            Tuple[bool, str]: Success flag and object name
        """
        try:
            # Get appropriate folder based on content type
            folder_path = self._get_folder_path(content_type, is_extracted_image)

            # Add document_id subfolder for extracted content if provided
            if is_extracted_image and document_id:
                folder_path += f"{document_id}/"

            # Generate unique object name with folder prefix
            ext = os.path.splitext(filename)[1]
            object_id = str(uuid.uuid4())
            object_name = f"{folder_path}{object_id}{ext}"

            # Ensure metadata is dict
            if metadata is None:
                metadata = {}

            # Add folder location to metadata
            metadata["folder"] = folder_path
            if document_id:
                metadata["document_id"] = document_id

            # Upload to MinIO
            self.client.put_object(
                bucket_name=MINIO_BUCKET_NAME,
                object_name=object_name,
                data=io.BytesIO(file_content),
                length=len(file_content),
                content_type=content_type,
                metadata=metadata,
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
                bucket_name=MINIO_BUCKET_NAME, object_name=object_name, expires=expires
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
                bucket_name=MINIO_BUCKET_NAME, object_name=object_name
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
                bucket_name=MINIO_BUCKET_NAME, object_name=object_name
            )
            return True
        except S3Error as e:
            print(f"Error deleting file: {e}")
            return False

    def list_objects(self, prefix: str = "") -> List[Dict[str, Any]]:
        """
        List objects in the bucket with optional prefix (folder)

        Args:
            prefix: Optional folder prefix to filter results

        Returns:
            List[Dict[str, Any]]: List of objects with metadata
        """
        try:
            objects = self.client.list_objects(
                MINIO_BUCKET_NAME, prefix=prefix, recursive=True
            )
            result = []

            for obj in objects:
                result.append(
                    {
                        "name": obj.object_name,
                        "size": obj.size,
                        "last_modified": obj.last_modified,
                    }
                )

            return result
        except S3Error as e:
            print(f"Error listing objects: {e}")
            return []

    def list_folder_contents(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        List contents of a specific folder

        Args:
            folder_path: Folder path (e.g., "documents/pdf/")

        Returns:
            List of objects in the folder
        """
        return self.list_objects(prefix=folder_path)

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
            return temp_file.name
