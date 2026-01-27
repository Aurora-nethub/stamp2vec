"""
Main FastAPI application for Seal Embedding Service
"""

import os
import json
import torch
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from train.model import SealEmbeddingNet
from seal_embedding_api.core.embedding_service import EmbeddingService
from seal_embedding_api.core.detection_service import DetectionService
from seal_embedding_api.core.storage import Storage
from seal_embedding_api.core.similarity_service import SimilarityService
from seal_embedding_api.api import health, pipeline


# Global state
app_state = {
    "model": None,
    "config": None,
    "embedding_service": None,
    "detection_service": None,
    "storage": None,
    "similarity_service": None,
    "device": None,
}


def init_app() -> FastAPI:
    """Initialize FastAPI application"""
    
    app = FastAPI(
        title="Seal Embedding API",
        description="API for seal detection, embedding extraction, and similarity search",
        version="0.1.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize model and services on startup"""
        try:
            print("[INFO] Initializing Seal Embedding API...")
            
            # Determine device (CPU only)
            device = torch.device("cpu")
            app_state["device"] = device
            print(f"[INFO] Using device: {device}")
            
            # Load model and config
            model_pkg_dir = "models/seal_pkg_v1"
            if not os.path.exists(model_pkg_dir):
                raise FileNotFoundError(f"Model package not found: {model_pkg_dir}")
            
            cfg_path = os.path.join(model_pkg_dir, "config.json")
            with open(cfg_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print(f"[INFO] Model config loaded from {cfg_path}")
            print(f"[INFO] Embedding dimension: {config.get('embedding_dim')}")
            
            # Load model using from_package method
            model, cfg = SealEmbeddingNet.from_package(
                pkg_dir=model_pkg_dir,
                device=device,
                verbose=True
            )
            
            app_state["model"] = model
            app_state["config"] = cfg
            
            # Initialize services
            app_state["embedding_service"] = EmbeddingService(model, cfg, device)
            app_state["detection_service"] = DetectionService()
            app_state["storage"] = Storage(base_dir="database/embeddings")
            app_state["similarity_service"] = SimilarityService()
            
            print("[INFO] All services initialized successfully!")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize: {str(e)}")
            raise
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown"""
        print("[INFO] Shutting down Seal Embedding API...")
        # Add any cleanup code here if needed
    
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
    import uvicorn
    
    # Run server
    uvicorn.run(
        "seal_embedding_api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
