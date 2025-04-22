import os
import uuid
from typing import Any, Dict, List, Optional

# Set environment variable to disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer

import src.config.env as env
from app.services.file_extractor_service import PDFTextExtractor, extract_text_from_file
from src.app.services.storage_service import StorageService
from app.services.vector_database_service import VectorDatabaseService
from src.app.services.database_service import DatabaseService


class EmbeddingService:
    """Service for text chunking, embedding generation, and vector database storage."""

    def __init__(self):
        """Initialize the embedding model and vector database service."""
        self.model = SentenceTransformer(env.EMBEDDING_MODEL)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        self.vector_db = VectorDatabaseService()
        self.storage_service = StorageService()

    def get_dimension(self) -> int:
        """Get the embedding dimension of the model."""
        return self.embedding_dimension

    def split_text_into_chunks(
        self,
        text: str,
        chunk_size: int = env.DEFAULT_CHUNK_SIZE,
        overlap: int = env.DEFAULT_CHUNK_OVERLAP,
    ) -> List[str]:
        """
        Split text into overlapping chunks with minimal whitespace, ensuring the chunks are tightly packed.

        Args:
            text: Input text to split.
            chunk_size: Maximum size of each chunk.
            overlap: Number of overlapping words between chunks.

        Returns:
            List of text chunks.
        """
        if not text:
            return []

        # Normalize the text: remove extra spaces and line breaks
        text = " ".join(
            text.split()
        )  # Replace multiple spaces and newlines with a single space

        chunks = []
        current_chunk = ""

        words = text.split()  # Split into words for easier management
        for word in words:
            # Check if the current chunk with this word exceeds the max chunk size
            if len(current_chunk) + len(word) + 1 > chunk_size and current_chunk:
                chunks.append(
                    current_chunk.strip()
                )  # Add the current chunk to the list
                current_chunk = word  # Start a new chunk with the current word
            else:
                if current_chunk:
                    current_chunk += " " + word  # Add the word to the current chunk
                else:
                    current_chunk = word  # Start with the first word

        if current_chunk.strip():
            chunks.append(current_chunk.strip())  # Add the last chunk if it's not empty

        # Optionally handle overlap (this could be adjusted if overlap needs to be stricter)
        if overlap > 0:
            chunks = [
                chunks[i] + " " + chunks[i + 1][:overlap]
                for i in range(len(chunks) - 1)
            ]

        return chunks

    def split_text_by_page(
        self,
        text_by_page: Dict[int, str],
        chunk_size: int = env.DEFAULT_CHUNK_SIZE,
        overlap: int = env.DEFAULT_CHUNK_OVERLAP,
    ) -> List[Dict[str, Any]]:
        """
        Split text by page into chunks while tracking page numbers.

        Args:
            text_by_page: Dictionary mapping page numbers to text content
            chunk_size: Maximum size of each chunk
            overlap: Number of overlapping words between chunks

        Returns:
            List of dictionaries containing chunk text and page info
        """
        chunks_with_page = []

        for page_num, page_text in text_by_page.items():
            if not page_text.strip():
                continue

            # Split this page's text into chunks
            page_chunks = self.split_text_into_chunks(page_text, chunk_size, overlap)

            # Add page info to each chunk
            for chunk_text in page_chunks:
                chunks_with_page.append(
                    {
                        "text": chunk_text,
                        "page_number": page_num + 1,  # Convert to 1-based page numbers
                    }
                )

        return chunks_with_page

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,
        ).tolist()
        return embeddings

    def create_embeddings(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        store: bool = False,
    ) -> Dict[str, Any]:
        """
        Embed texts and optionally store in vector database.

        Args:
            texts: List of text strings to embed.
            metadata: Optional metadata for each text.
            store: Whether to store embeddings in vector DB.

        Returns:
            Dictionary with embedding information and optional vector IDs.
        """
        embeddings = self.embed_texts(texts)
        result = {
            "count": len(embeddings),
            "dimension": self.embedding_dimension,
        }

        if store:
            if not metadata or len(metadata) != len(texts):
                metadata_list = [
                    {"text": text, "id": str(uuid.uuid4())} for text in texts
                ]
            else:
                metadata_list = metadata
                for i, md in enumerate(metadata_list):
                    if "text" not in md:
                        md["text"] = texts[i]

            ids = [md.get("id", str(uuid.uuid4())) for md in metadata_list]
            stored_ids = self.vector_db.store_vectors(embeddings, metadata_list, ids)
            result["vector_ids"] = stored_ids
        else:
            result["embeddings"] = embeddings

        return result

    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        chunk_size: int = env.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = env.DEFAULT_CHUNK_OVERLAP,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a document: extract text, chunk it, embed chunks, and store in vector DB.
        Now with enhanced PDF processing including images.

        Args:
            file_content: Binary content of the file.
            filename: Name of the file.
            content_type: MIME type of the file.
            chunk_size: Maximum size of each chunk.
            chunk_overlap: Number of overlapping words between chunks.
            base_metadata: Optional base metadata for the document.

        Returns:
            Dictionary with processing results including chunk information and vector IDs.
        """
        try:
            # Initialize result structure
            result = {
                "filename": filename,
                "chunks": [],
                "vector_ids": [],
                "file_id": "",
                "images": [],
            }

            # Initialize basic metadata
            if base_metadata is None:
                base_metadata = {}
            base_metadata["active"] = True

            document_id = base_metadata.get("file_id", str(uuid.uuid4()))
            base_metadata["file_id"] = document_id
            result["file_id"] = document_id

            print(f"Processing document ID: {document_id}")

            # Special handling for PDF files with enhanced extraction
            if "pdf" in content_type.lower():
                return await self.process_pdf_document(
                    file_content=file_content,
                    filename=filename,
                    content_type=content_type,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    base_metadata=base_metadata,
                )

            # Regular text extraction for non-PDF files
            text = extract_text_from_file(file_content, content_type)
            if not text or len(text.strip()) < 10:
                print(f"Warning: Insufficient text extracted from file: {filename}")
                raise ValueError("Insufficient text extracted from file")

            # Split into chunks
            chunks = self.split_text_into_chunks(text, chunk_size, chunk_overlap)
            if not chunks:
                raise ValueError("No text chunks generated")

            print(f"Processing document ID: {document_id} with {len(chunks)} chunks")

            db_service = DatabaseService()

            # Process each chunk
            vector_ids = []
            chunk_results = []
            for i, chunk_text in enumerate(chunks):
                try:
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata.update(
                        {
                            "chunk_index": i,
                            "filename": filename,
                            "content_type": content_type,
                            "text": chunk_text,
                        }
                    )

                    print(f"Saving chunk {i} to database")
                    chunk = db_service.create_document_chunk(
                        document_id=document_id,
                        chunk_index=i,
                        text=chunk_text,
                        metadata=chunk_metadata,
                    )

                    print(f"Creating embedding for chunk {i}")
                    embedding_result = self.create_embeddings([chunk_text])
                    if (
                        "embeddings" in embedding_result
                        and embedding_result["embeddings"]
                    ):
                        embedding = embedding_result["embeddings"][0]
                        vector_id = str(uuid.uuid4())
                        print(f"Storing vector with ID: {vector_id}")
                        self.vector_db.store_vectors(
                            [embedding], [chunk_metadata], [vector_id]
                        )
                        vector_ids.append(vector_id)

                        db_service.update_chunk_embedding(chunk.id, vector_id)
                        print(f"Chunk {i} processed with vector ID: {vector_id}")

                        chunk_results.append(
                            {"text": chunk_text, "metadata": chunk_metadata}
                        )
                    else:
                        print(f"Warning: No embedding generated for chunk {i}")

                except Exception as chunk_error:
                    print(f"Error processing chunk {i}: {str(chunk_error)}")
                    import traceback

                    traceback.print_exc()

            result["chunks"] = chunk_results
            result["vector_ids"] = vector_ids

            return result

        except Exception as e:
            print(f"Error in process_document: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

    async def process_pdf_document(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        chunk_size: int = env.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = env.DEFAULT_CHUNK_OVERLAP,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a PDF document with enhanced extraction of text and images.

        Args:
            file_content: Binary content of the file.
            filename: Name of the file.
            content_type: MIME type of the file.
            chunk_size: Maximum size of each chunk.
            chunk_overlap: Number of overlapping words between chunks.
            base_metadata: Optional base metadata for the document.

        Returns:
            Dictionary with processing results including chunk and image information.
        """
        try:
            # Use enhanced PDF extractor
            pdf_extractor = PDFTextExtractor(file_content)
            extraction_result = pdf_extractor.extract_text()

            # Initialize result structure
            result = {
                "filename": filename,
                "chunks": [],
                "vector_ids": [],
                "file_id": "",
                "images": [],
            }

            # Get document ID from metadata
            document_id = base_metadata.get("file_id", str(uuid.uuid4()))
            base_metadata["file_id"] = document_id
            result["file_id"] = document_id

            # Initialize database service
            db_service = DatabaseService()

            # Extract text by page and create chunks
            if "text_by_page" in extraction_result:
                text_by_page = extraction_result["text_by_page"]
                chunks_with_page = self.split_text_by_page(
                    text_by_page, chunk_size, chunk_overlap
                )

                # Process chunks with page information
                vector_ids = []
                chunk_results = []

                for i, chunk_info in enumerate(chunks_with_page):
                    try:
                        chunk_text = chunk_info["text"]
                        page_number = chunk_info["page_number"]

                        # Prepare chunk metadata
                        chunk_metadata = base_metadata.copy()
                        chunk_metadata.update(
                            {
                                "chunk_index": i,
                                "page_number": page_number,
                                "filename": filename,
                                "content_type": content_type,
                                "text": chunk_text,
                            }
                        )

                        # Find related images for this page
                        related_images = []
                        for img in extraction_result.get("images", []):
                            if img.get("page_number") == page_number:
                                related_images.append(str(img.get("id", "")))

                        # Add references to related elements
                        if related_images:
                            chunk_metadata["related_images"] = related_images

                        # Save chunk to database
                        print(f"Saving chunk {i} (page {page_number}) to database")
                        chunk = db_service.create_document_chunk(
                            document_id=document_id,
                            chunk_index=i,
                            text=chunk_text,
                            page_number=page_number,
                            metadata=chunk_metadata,
                            related_images=related_images,
                        )

                        # Create embedding
                        print(f"Creating embedding for chunk {i}")
                        embedding_result = self.create_embeddings([chunk_text])
                        if (
                            "embeddings" in embedding_result
                            and embedding_result["embeddings"]
                        ):
                            embedding = embedding_result["embeddings"][0]
                            vector_id = str(uuid.uuid4())
                            print(f"Storing vector with ID: {vector_id}")
                            self.vector_db.store_vectors(
                                [embedding], [chunk_metadata], [vector_id]
                            )
                            vector_ids.append(vector_id)

                            db_service.update_chunk_embedding(chunk.id, vector_id)
                            print(f"Chunk {i} processed with vector ID: {vector_id}")

                            chunk_results.append(
                                {
                                    "text": chunk_text,
                                    "metadata": chunk_metadata,
                                    "page_number": page_number,
                                }
                            )
                        else:
                            print(f"Warning: No embedding generated for chunk {i}")

                    except Exception as chunk_error:
                        print(f"Error processing chunk {i}: {str(chunk_error)}")
                        import traceback

                        traceback.print_exc()

                result["chunks"] = chunk_results
                result["vector_ids"] = vector_ids

            # Process images - now using organized storage
            stored_images = []
            for img_index, img in enumerate(extraction_result.get("images", [])):
                try:
                    if not img.get("data"):
                        continue

                    # Store image in MinIO with organized folders
                    image_storage_path = None
                    try:
                        if img.get("data"):
                            # Create a file name for the image
                            img_ext = img.get("format", "png")
                            img_filename = f"{filename.split('.')[0]}_page{img.get('page_number')}_img{img_index}.{img_ext}"

                            # Upload image to MinIO using the organized storage service
                            success, image_storage_path = (
                                await self.storage_service.upload_file(
                                    img["data"],
                                    img_filename,
                                    f"image/{img_ext}",
                                    {
                                        "document_id": document_id,
                                        "source": "pdf_extraction",
                                    },
                                    is_extracted_image=True,  # Indicate this is an extracted image
                                    document_id=document_id,  # Pass document_id for folder organization
                                )
                            )

                            if not success:
                                print(f"Failed to upload image {img_index} to storage")
                                image_storage_path = None
                    except Exception as upload_error:
                        print(f"Error uploading image to storage: {upload_error}")

                    # Create image record in database
                    image = db_service.create_document_image(
                        document_id=document_id,
                        page_number=img.get("page_number", 0),
                        image_index=img.get("image_index", img_index),
                        width=img.get("width"),
                        height=img.get("height"),
                        format=img.get("format"),
                        storage_path=image_storage_path,
                        ocr_text=img.get("ocr_text"),
                        metadata={
                            "extracted_from": filename,
                            "source": "pdf_extraction",
                        },
                    )

                    # If image has OCR text, create embeddings for it
                    if img.get("ocr_text"):
                        ocr_text = img["ocr_text"]

                        # Create metadata for the OCR text
                        ocr_metadata = base_metadata.copy()
                        ocr_metadata.update(
                            {
                                "source": "image_ocr",
                                "image_id": image.id,
                                "page_number": img.get("page_number", 0),
                                "filename": filename,
                                "text": ocr_text,
                                "storage_path": image_storage_path,
                            }
                        )

                        # Create embedding for the OCR text
                        ocr_embedding_result = self.create_embeddings([ocr_text])
                        if (
                            "embeddings" in ocr_embedding_result
                            and ocr_embedding_result["embeddings"]
                        ):
                            ocr_embedding = ocr_embedding_result["embeddings"][0]
                            ocr_vector_id = str(uuid.uuid4())
                            self.vector_db.store_vectors(
                                [ocr_embedding], [ocr_metadata], [ocr_vector_id]
                            )

                    # Add to stored images list
                    img["id"] = image.id  # Add database ID to the result
                    img["storage_path"] = image_storage_path
                    stored_images.append(img)

                except Exception as img_error:
                    print(f"Error processing image {img_index}: {str(img_error)}")

            result["images"] = stored_images

            return result

        except Exception as e:
            print(f"Error in process_pdf_document: {str(e)}")
            import traceback

            traceback.print_exc()
            raise
