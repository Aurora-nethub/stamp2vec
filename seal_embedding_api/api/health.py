"""
Health check endpoint
"""

import time
from pathlib import Path
from typing import List
from fastapi import APIRouter, Request, HTTPException
from PIL import Image
import numpy as np

from ..models import HealthCheckResponse, VerifySummary


router = APIRouter()


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _list_images(directory: Path) -> List[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted([p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_EXTS])


async def _run_verify(embedding_service, query_dir: Path, candidate_dir: Path, max_images: int):
    query_files = _list_images(query_dir)[:max_images]
    candidate_files = _list_images(candidate_dir)[:max_images]

    if not query_files or not candidate_files:
        raise HTTPException(status_code=400, detail="Verification data not found")

    query_images = [Image.open(p).convert("RGB") for p in query_files]
    candidate_images = [Image.open(p).convert("RGB") for p in candidate_files]

    q_emb = await embedding_service.extract_embeddings_batch(query_images)
    c_emb = await embedding_service.extract_embeddings_batch(candidate_images)

    if q_emb.size == 0 or c_emb.size == 0:
        raise HTTPException(status_code=500, detail="Failed to extract embeddings")

    sims = q_emb @ c_emb.T
    top1_scores = np.max(sims, axis=1)

    return {
        "query_count": len(query_files),
        "candidate_count": len(candidate_files),
        "top1_avg": float(np.mean(top1_scores)),
        "top1_min": float(np.min(top1_scores)),
        "top1_max": float(np.max(top1_scores)),
    }


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    fastapi_request: Request,
    verify: bool = False,
    max_images: int = 3,
    force: bool = False,
    verify_ttl_sec: int = 300,
):
    """
    Health check endpoint - tests if model is loaded and can run embedding
    """
    try:
        app_state = fastapi_request.app.state
        embedding_service = app_state.embedding_service

        if embedding_service is None:
            return HealthCheckResponse(
                status="unhealthy",
                model_loaded=False,
                embedding_dim=768,
                message="Embedding service not initialized",
            )

        response = HealthCheckResponse(
            status="healthy",
            model_loaded=True,
            embedding_dim=768,
            message="Model is ready",
        )

        if not verify:
            return response

        now = time.time()
        cache = getattr(app_state, "health_verify_cache", None)
        if cache and not force and now - cache["ts"] < verify_ttl_sec:
            response.verify = VerifySummary(**cache["summary"], cached=True)
            return response

        start = time.time()
        summary = await _run_verify(
            embedding_service=embedding_service,
            query_dir=Path("data/query"),
            candidate_dir=Path("data/candidate"),
            max_images=max_images,
        )
        duration_ms = int((time.time() - start) * 1000)

        verify_summary = VerifySummary(
            **summary,
            duration_ms=duration_ms,
            cached=False,
        )
        app_state.health_verify_cache = {
            "ts": now,
            "summary": verify_summary.dict(),
        }
        response.verify = verify_summary
        return response
    except HTTPException as e:
        return HealthCheckResponse(
            status="unhealthy",
            model_loaded=False,
            embedding_dim=768,
            message=str(e.detail),
        )
    except Exception as e:
        return HealthCheckResponse(
            status="unhealthy",
            model_loaded=False,
            embedding_dim=768,
            message=str(e),
        )
