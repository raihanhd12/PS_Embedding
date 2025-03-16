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
├── utils/                    # Utilities
│   ├── __init__.py
│   └── file_utils.py         # File handling utilities
└── scripts/                  # Utility scripts
    ├── __init__.py
    ├── init_db.py            # Database initialization script
    └── reset_db.py           # Database reset script
```

## Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL (optional, can be run in Docker)
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

2. Start the required services using Docker Compose:

   ```
   docker-compose up -d qdrant elasticsearch kibana minio
   ```

3. Initialize databases (optional):

   ```
   python scripts/init_db.py
   ```

4. Start the API server:

   ```
   uvicorn app:app --reload
   ```

5. Access the API documentation at http://localhost:8000/docs

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

- `POST /api/upload/batch` - Upload multiple documents in a single request

  - Form data:
    - `files`: List of document files (PDF, DOCX, TXT)
    - `metadata`: JSON string with additional metadata (optional)

- `POST /api/embedding/batch` - Process and embed multiple documents
  - Request body:
    ```json
    {
      "file_ids": ["id1", "id2", ...],
      "chunk_size": 1000,
      "chunk_overlap": 200,
      "additional_metadata": {"key": "value"}
    }
    ```

### Vector Search

- `POST /api/search` - Search for similar text using hybrid search (vector + keyword)
  - Request body:
    ```json
    {
      "query": "your search query",
      "limit": 5,
      "filter_metadata": { "key": "value" }
    }
    ```

### Document Management

- `GET /api/documents` - Get a list of all documents

  - Query parameters:
    - `limit`: Maximum number of documents (default: 100)
    - `offset`: Pagination offset (default: 0)

- `GET /api/documents/{document_id}` - Get a specific document by ID

- `DELETE /api/documents/batch` - Delete multiple documents and all associated data
  - Request body:
    ```json
    ["document_id1", "document_id2", ...]
    ```

## Hybrid Search

The API supports hybrid search that combines vector similarity (embeddings) and keyword matching (Elasticsearch) for improved search results. The weights between vector and keyword search can be configured in the `.env` file:

```
ENABLE_HYBRID_SEARCH=True
VECTOR_WEIGHT=0.7
KEYWORD_WEIGHT=0.3
```

## Admin Interfaces

- Qdrant Dashboard: http://localhost:6333/dashboard
- Kibana (Elasticsearch): http://localhost:5601
- MinIO Console: http://localhost:9001
  - Username: minioadmin
  - Password: minioadmin
