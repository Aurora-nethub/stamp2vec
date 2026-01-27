"""
Pydantic models for API request/response
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Status: 'healthy' or 'unhealthy'")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    embedding_dim: int = Field(..., description="Embedding dimension")
    message: str = Field(..., description="Status message")


class EmbeddingMetadata(BaseModel):
    """Metadata for a stored embedding"""
    seal_id: str
    doc_id: str
    embedding_dim: int
    image_crop_saved: bool


class PipelineRequest(BaseModel):
    """Request for detect_and_embed pipeline"""
    image: str = Field(..., description="Base64-encoded image")
    doc_id: str = Field(..., description="Document ID for association")
    save_crops: bool = Field(default=True, description="Whether to save cropped seal images")


class PipelineResponse(BaseModel):
    """Response from detect_and_embed pipeline"""
    doc_id: str
    seals_detected: int
    embeddings_stored: List[EmbeddingMetadata]
    status: str = "success"


class SimilarityMatch(BaseModel):
    """A single similarity match result"""
    seal_id: str
    doc_id: str
    similarity_score: float
    rank: int


class SimilaritySearchRequest(BaseModel):
    """Request for similarity search"""
    image: Optional[str] = Field(None, description="Base64-encoded image (for new image search)")
    query_seal_id: Optional[str] = Field(None, description="Existing seal ID to search from database")
    top_k: int = Field(default=3, description="Number of top matches to return")

    class Config:
        validate_assignment = True

    def __init__(self, **data):
        super().__init__(**data)
        if not self.image and not self.query_seal_id:
            raise ValueError("Either 'image' or 'query_seal_id' must be provided")


class SimilaritySearchResponse(BaseModel):
    """Response from similarity search"""
    query_seal_id: str
    matches: List[SimilarityMatch]
    total_in_database: int
    returned_count: int
    status: str = "success"
