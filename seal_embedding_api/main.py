from contextlib import asynccontextmanager
import os
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from seal_embedding_api.core.seal_model import SealEmbeddingNet
from seal_embedding_api.core.embedding_service import EmbeddingService
from seal_embedding_api.core.detection_service import DetectionService
from seal_embedding_api.core.milvus_service import MilvusService
from seal_embedding_api.config_loader import ConfigLoader
from seal_embedding_api.logger_config import get_logger
from seal_embedding_api.api import health, pipeline


def init_app() -> FastAPI:
    logger = get_logger(__name__)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            logger.info("Initializing Seal Embedding API...")

            config = ConfigLoader.load("config/api_config.json")

            device = torch.device(config.embedding_model.device)
            logger.info(f"Using device: {device}")

            model, cfg = SealEmbeddingNet.from_package(
                pkg_dir=config.embedding_model.pkg_dir,
                device=device,
                verbose=True
            )

            app.state.model = model
            app.state.config = config
            app.state.device = device

            app.state.embedding_service = EmbeddingService(
                model,
                cfg,
                device,
                batch_size=config.batch_processing.embedding_batch_size,
            )
            app.state.detection_service = DetectionService(config.detection_model)

            milvus_db_path = config.milvus.db_path
            if not os.path.isabs(milvus_db_path):
                milvus_db_path = os.path.join(os.getcwd(), milvus_db_path)
            
            app.state.milvus_service = MilvusService(
                db_path=milvus_db_path,
                collection_name=config.milvus.collection_name,
            )

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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(pipeline.router)

    return app


app = init_app()


if __name__ == "__main__":
    uvicorn.run(
        "seal_embedding_api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
