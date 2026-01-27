"""
Pipeline endpoints for detection, embedding, and similarity search
"""

from fastapi import APIRouter, Depends
from PIL import Image
import io
import base64
from typing import Optional

from ..models import (
    PipelineRequest, PipelineResponse, EmbeddingMetadata,
    SimilaritySearchRequest, SimilaritySearchResponse, SimilarityMatch
)


router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post("/detect_and_embed", response_model=PipelineResponse)
async def detect_and_embed(request: PipelineRequest):
    """
    Detect seals in image and extract embeddings
    
    Steps:
    1. Decode base64 image
    2. Detect seals using LayoutDetection
    3. Extract embeddings for each seal
    4. Store embeddings and metadata
    5. Return results
    """
    # TODO: Implement
    pass


@router.post("/similarity_search", response_model=SimilaritySearchResponse)
async def similarity_search(request: SimilaritySearchRequest):
    """
    Search for similar seals in database
    
    Input options:
    - image: New seal image (will extract embedding)
    - query_seal_id: Existing seal ID from database
    
    Returns:
    - Top-k similar seals from database
    - If database has < k items, return all
    """
    # TODO: Implement
    pass
