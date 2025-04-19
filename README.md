# Embedding API

A modular API service for text embedding and semantic search with PostgreSQL, Qdrant, and MinIO integration.

## Features

- Generate embeddings for text input
- Process and extract text from PDF and DOCX files
- Automatically chunk documents for optimal embedding
- Store document metadata in PostgreSQL
- Store vectors in Qdrant for efficient similarity search
- Store original documents in MinIO
- Clean, maintainable code structure with environment-based configuration

## System Requirements

- Python 3.8+
- PostgreSQL
- Qdrant vector database
- MinIO object storage
- Poppler (for PDF processing with pdf2image)
- Tesseract OCR (for text extraction from images and scanned documents)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/raihanhd12/PS_Embedding.git
cd PS_Embedding
```

### 2. Set up a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Poppler and Tesseract

#### For Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils
sudo apt-get install -y tesseract-ocr
```

#### For macOS:

```bash
brew install poppler
brew install tesseract
```

#### For Windows:

- Download and install [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)
- Download and install [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki)
- Add both to your PATH environment variable

### 5. Configure environment variables

Copy the example environment file and update it with your settings:

```bash
cp .env.example .env
```

Edit the `.env` file with your configuration:

## Running with Docker

### 1. Start the required services

```bash
docker-compose up -d
```

This will start:

- Qdrant (vector database) - available at http://localhost:6333
- MinIO (document storage) - available at http://localhost:9000 (API) and http://localhost:9001 (Console)

### 2. Initialize the databases

```bash
python scripts/migrate_fresh.py --confirm
```

### 3. Start the API server

```bash
python main.py
```

The API will be available at http://localhost:8001 with API documentation at http://localhost:8001/docs.

## API Endpoints

### Authentication

All endpoints require an API key to be sent in the `X-API-Key` header.

### Document Management

- `POST /api/v1/upload/batch`: Upload multiple documents
- `POST /api/v1/embedding/batch`: Process and embed documents from file IDs
- `POST /api/v1/embedding/local`: Process and embed local files
- `GET /api/v1/documents`: List all documents
- `GET /api/v1/documents/{document_id}`: Get a specific document
- `DELETE /api/v1/documents/batch`: Delete multiple documents
- `DELETE /api/v1/documents/local/batch`: Delete multiple local documents
- `POST /api/v1/documents/{document_id}/toggle-status`: Enable/disable a document
- `POST /api/v1/search`: Search for semantically similar content

## Data Flow Architecture

1. **Document Upload**: Files are uploaded and stored in MinIO object storage.
2. **Text Extraction**: Text is extracted from documents (PDF, DOCX, etc.).
3. **Chunking**: Documents are split into manageable chunks with overlap.
4. **Embedding Generation**: Each chunk is processed by the embedding model.
5. **Vector Storage**: Embeddings are stored in Qdrant for similarity search.
6. **Metadata Storage**: Document and chunk metadata are stored in PostgreSQL.
7. **Search**: Queries are embedded and similarity search is performed in Qdrant.

## Maintenance

### Reset All Data

```bash
python scripts/migrate_fresh.py --confirm
```

### Update the Vector Database

To change the embedding model or modify the vector size:

1. Update the `.env` file with the new settings
2. Reset the vector database: `python scripts/migrate_fresh.py --confirm`
3. Re-embed your documents

## Troubleshooting

- **Connection issues**: Ensure all services (PostgreSQL, Qdrant, MinIO) are running
- **Authentication failures**: Verify API key in the `.env` file and request headers
- **Embedding errors**: Check for sufficient disk space and memory
- **PDF extraction fails**: Verify Poppler is correctly installed and accessible
- **OCR not working**: Ensure Tesseract is properly installed and in your PATH

## License

MIT
