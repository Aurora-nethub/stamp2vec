"""
Main FastAPI application for Seal Embedding Service
"""

from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from train.model import SealEmbeddingNet
from seal_embedding_api.core.embedding_service import EmbeddingService
from seal_embedding_api.core.detection_service import DetectionService
from seal_embedding_api.core.storage import Storage
from seal_embedding_api.core.similarity_service import SimilarityService
from seal_embedding_api.config_loader import ConfigLoader
from seal_embedding_api.logger_config import get_logger
from seal_embedding_api.api import health, pipeline


def init_app() -> FastAPI:
    """Initialize FastAPI application"""
    logger = get_logger(__name__)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize and cleanup resources"""
        try:
            logger.info("Initializing Seal Embedding API...")

            # Load config
            config = ConfigLoader.load("config/api_config.json")

            # Determine device (CPU only)
            device = torch.device("cpu")
            logger.info(f"Using device: {device}")

            # Load model using from_package method
            model, cfg = SealEmbeddingNet.from_package(
                pkg_dir=config.embedding_model.pkg_dir,
                device=device,
                verbose=True
            )

            app.state.model = model
            app.state.config = config
            app.state.device = device

            # Initialize services
            app.state.embedding_service = EmbeddingService(
                model,
                cfg,
                device,
                batch_size=config.batch_processing.embedding_batch_size,
            )
            app.state.detection_service = DetectionService(config.detection_model)
            app.state.storage = Storage(base_dir=config.storage.base_dir)
            app.state.similarity_service = SimilarityService()

            logger.info("All services initialized successfully!")
            yield
        except Exception as e:
            logger.error(f"Failed to initialize: {str(e)}", exc_info=True)
            raise
        finally:
            logger.info("Shutting down Seal Embedding API...")

    app = FastAPI(
        title="Seal Embedding API",
        description="API for seal detection, embedding extraction, and similarity search",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router)
    app.include_router(pipeline.router)

    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "service": "Seal Embedding API",
            "status": "running",
            "version": "0.1.0"
        }

    return app


# Create app instance
app = init_app()


if __name__ == "__main__":
    
    # Run server
    uvicorn.run(
        "seal_embedding_api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
