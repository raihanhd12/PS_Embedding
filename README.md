# Embedding API

A modular API service for text embedding and semantic search with PostgreSQL, Qdrant, Elasticsearch, and MinIO integration.

## Features

- Generate embeddings for text input
- Process and extract text from PDF and DOCX files
- Automatically chunk documents for optimal embedding
- Store document metadata in PostgreSQL
- Store vectors in Qdrant for efficient similarity search
- Index text in Elasticsearch for keyword search
- Store original documents in MinIO
- Support for hybrid search (combining vector and keyword search)
- Clean, maintainable code structure with environment-based configuration

## Project Structure

```
embedding_api/
├── app.py                    # Main application entry point
├── config.py                 # Configuration settings
├── requirements.txt          # Dependencies
├── .env                      # Environment variables
├── .env.example              # Environment variables template
├── Dockerfile                # Dockerfile for containerization
├── docker-compose.yml        # Docker Compose configuration
├── models/                   # Data models
│   ├── __init__.py
│   └── schemas.py            # Pydantic models for API
├── services/                 # Core services
│   ├── __init__.py
│   ├── database.py           # PostgreSQL database service
│   ├── embedding.py          # Text embedding service
│   ├── vector_db.py          # Qdrant vector database service
│   ├── elasticsearch.py      # Elasticsearch service
│   ├── storage.py            # MinIO storage service
│   └── text_processor.py     # Text extraction and chunking
├── api/                      # API routes
│   ├── __init__.py
│   └── endpoints.py          # API endpoints
└── utils/                    # Utilities
    ├── __init__.py
    └── file_utils.py         # File handling utilities
```

## Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL
- Qdrant
- Elasticsearch
- MinIO
- Docker and Docker Compose (optional, for containerized setup)

### Environment Setup

1. Clone this repository
2. Create a `.env` file based on `.env.example`:
   ```
   cp .env.example .env
   ```
3. Edit the `.env` file to match your environment

### Local Development Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start the required services:
   - PostgreSQL database
   - Qdrant vector database
   - Elasticsearch
   - MinIO

3. Start the API server:
   ```
   uvicorn app:app --reload
   ```

4. Access the API documentation at http://localhost:8000/docs

### Docker Setup

1. Build and start all services using Docker Compose:
   ```
   docker-compose up -d
   ```

2. Access the API documentation at http://localhost:8000/docs

## API Endpoints

### Text Embedding

- `POST /api/embed` - Generate embeddings for text
  - Request body:
    ```json
    {
      "texts": ["text1", "text2", ...],
      "metadata": [{"key": "value"}, ...],
      "store": true
    }
    ```

### Document Processing

- `POST /api/upload-embed` - Upload, process, and embed document
  - Form data:
    - `file`: Document file (PDF, DOCX, TXT)
    - `chunk_size`: Size of text chunks (optional)
    - `chunk_overlap`: Overlap between chunks (optional)
    - `metadata`: JSON string with additional metadata (optional)

### Vector Search

- `POST /api/search` - Search for similar text
  - Request body:
    ```json
    {
      "query": "your search query",
      "limit": 5,
      "filter_metadata": {"key": "value"}
    }
    ```

### Document Management

- `DELETE /api/vectors/{vector_id}` - Delete specific vector
- `DELETE /api/documents/{file_id}` - Delete document and all associated vectors