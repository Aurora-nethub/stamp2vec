from typing import List, Optional
from pydantic import BaseModel, model_validator


class VerifySummary(BaseModel):
    query_count: int
    candidate_count: int
    query_detected: int
    candidate_detected: int
    top1_avg: float
    top1_min: float
    top1_max: float
    duration_ms: int
    cached: bool


class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    embedding_dim: int
    milvus_loaded: bool
    milvus_collection: str
    milvus_count: int
    message: str
    verify: Optional[VerifySummary] = None


class EmbeddingMetadata(BaseModel):
    seal_id: str
    embedding_dim: int
    crop_image_b64: str


class IngestItem(BaseModel):
    image_b64: str
    seal_id: str


class IngestRequest(BaseModel):
    items: List[IngestItem]


class IngestResponse(BaseModel):
    seals_detected: int
    embeddings_stored: List[EmbeddingMetadata]
    status: str = "success"


class SimilarityMatch(BaseModel):
    seal_id: str
    similarity_score: float
    rank: int
    image_base64: Optional[str] = None


class SimilaritySearchRequest(BaseModel):
    image_b64: Optional[str] = None
    query_seal_id: Optional[str] = None
    top_k: int = 3

    @model_validator(mode="after")
    def _check_top_k(self):
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        return self

    @model_validator(mode="after")
    def _check_query(self):
        if not self.image_b64 and not self.query_seal_id:
            raise ValueError("Either 'image_b64' or 'query_seal_id' must be provided")
        return self


class SimilaritySearchResponse(BaseModel):
    query_seal_id: str
    top_1: Optional[SimilarityMatch]
    top_3: List[SimilarityMatch]
    total_in_database: int
    returned_count: int
    image_base64: Optional[str] = None
    status: str = "success"


class DeleteRequest(BaseModel):
    seal_ids: List[str]


class DeleteResponse(BaseModel):
    deleted_count: int
    failed_count: int
    deleted_ids: List[str]
    failed_ids: List[str]
    status: str = "success"
