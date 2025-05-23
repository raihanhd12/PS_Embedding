from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import src.config.env as env
from src.app.middleware.session import session_middleware
from src.app.services.database_service import DatabaseService
from src.app.services.vector_database_service import VectorDatabaseService
from src.routes.api.v1 import router

# Initialize services for lifespan events
vector_db_service = VectorDatabaseService()
db_service = DatabaseService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    print("Starting up Embedding API...")

    # Initialize database (with error handling)
    try:
        db_service.init_db()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization warning: {e}")
        print("The application will attempt to continue")

    # Initialize vector database (with error handling)
    try:
        vector_db_service.init_vector_db()
        print("Vector database initialized successfully")
    except Exception as e:
        print(f"Vector database initialization warning: {e}")
        print("Some functionality may be limited")

    print("Embedding API initialized and ready")
    yield
    # Shutdown code (add here if needed in the future)
    print("Shutting down Embedding API")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Embedding API",
    description="API for text embedding and vector search",
    version="1.0.0",
    lifespan=lifespan,
)

# Add session middleware
app.middleware("http")(session_middleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Embedding API",
        "documentation": "/docs",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host=env.API_HOST, port=env.API_PORT)
