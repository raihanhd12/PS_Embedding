import os
import uuid
from typing import Any, Dict, List, Optional

# Set environment variable to disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import app.utils.config as config
from sentence_transformers import SentenceTransformer
from app.services.file_utils import extract_text_from_file
from app.services.vector_db import VectorDatabaseService


class EmbeddingService:
    """Service for text chunking, embedding generation, and vector database storage."""

    def __init__(self):
        """Initialize the embedding model and vector database service."""
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        self.vector_db = VectorDatabaseService()

    def get_dimension(self) -> int:
        """Get the embedding dimension of the model."""
        return self.embedding_dimension

    def split_text_into_chunks(
        self,
        text: str,
        chunk_size: int = config.DEFAULT_CHUNK_SIZE,
        overlap: int = config.DEFAULT_CHUNK_OVERLAP,
    ) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Input text to split.
            chunk_size: Maximum size of each chunk.
            overlap: Number of overlapping words between chunks.

        Returns:
            List of text chunks.
        """
        if not text:
            return []

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                words = current_chunk.split()
                if len(words) > overlap:
                    current_chunk = " ".join(words[-overlap:]) + "\n\n"
                else:
                    current_chunk = ""
            current_chunk += para + "\n\n"
            while len(current_chunk) > chunk_size:
                chunks.append(current_chunk[:chunk_size].strip())
                words = current_chunk[:chunk_size].split()
                if len(words) > overlap:
                    current_chunk = (
                        " ".join(words[-overlap:]) + "\n\n" + current_chunk[chunk_size:]
                    )
                else:
                    current_chunk = current_chunk[chunk_size:]
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks

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
        chunk_size: int = config.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a document: extract text, chunk it, embed chunks, and store in vector DB.

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
            text = extract_text_from_file(file_content, content_type)
            if not text or len(text.strip()) < 10:
                print(f"Warning: Insufficient text extracted from file: {filename}")
                raise ValueError("Insufficient text extracted from file")

            chunks = self.split_text_into_chunks(text, chunk_size, chunk_overlap)
            if not chunks:
                raise ValueError("No text chunks generated")

            if base_metadata is None:
                base_metadata = {}
            base_metadata["active"] = True

            document_id = base_metadata.get("file_id", str(uuid.uuid4()))
            base_metadata["file_id"] = document_id

            print(f"Processing document ID: {document_id} with {len(chunks)} chunks")

            from app.services.database import DatabaseService

            db_service = DatabaseService()

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

            return {
                "filename": filename,
                "chunks": chunk_results,
                "vector_ids": vector_ids,
                "file_id": document_id,
            }
        except Exception as e:
            print(f"Error in process_document: {str(e)}")
            import traceback

            traceback.print_exc()
            raise
