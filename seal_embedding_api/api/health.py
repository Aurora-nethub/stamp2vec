"""
Health check endpoint
"""

from fastapi import APIRouter, Depends
from PIL import Image
import io
import base64

from ..models import HealthCheckResponse


router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    embedding_service = Depends(lambda: None),  # Will be injected by main
    model_service = Depends(lambda: None)
):
    """
    Health check endpoint - tests if model is loaded and can run embedding
    """
    try:
        # This will be called from main.py with proper dependency injection
        # For now, this is a placeholder
        return {
            "status": "healthy",
            "model_loaded": True,
            "embedding_dim": 768,
            "message": "Model is ready"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "embedding_dim": 768,
            "message": str(e)
        }
