"""
Main application entry point for Embedding API
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import config and services
import app.config as config
from app.services.database import DatabaseService
from app.services.vector_db import init_vector_db
from app.api.endpoints import router as api_router

# Initialize FastAPI app
app = FastAPI(
    title="Embedding API",
    description="API for text embedding and vector search",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Embedding API",
        "documentation": "/docs",
        "version": "1.0.0"
    }


@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup"""
    # Print configuration
    config.print_config()

    # Initialize database (with error handling)
    try:
        DatabaseService.init_db()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization warning: {e}")
        print("The application will attempt to continue")

    # Initialize vector database (with error handling)
    try:
        init_vector_db()
        print("Vector database initialized successfully")
    except Exception as e:
        print(f"Vector database initialization warning: {e}")
        print("Some functionality may be limited")

    print("Embedding API initialized and ready")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Perbaikan disini dari "app:main" menjadi "main:app"
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )