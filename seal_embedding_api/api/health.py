import os
import time
import random
import threading
from pathlib import Path
from typing import List
from fastapi import APIRouter, Request, HTTPException
from PIL import Image
import numpy as np

from ..models import HealthCheckResponse, VerifySummary
from ..config_loader import ConfigLoader
from seal_embedding_api.core.seal_model import SealEmbeddingNet
from ..core.embedding_service import EmbeddingService
from ..core.detection_service import DetectionService
from ..core.milvus_service import MilvusService


router = APIRouter()


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _list_images(directory: Path) -> List[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted([p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_EXTS])


def _sample_files(files: List[Path], max_images: int, random_sample: bool) -> List[Path]:
    if max_images <= 0 or max_images >= len(files):
        return files
    if random_sample:
        return random.sample(files, max_images)
    return files[:max_images]


def _reinit_services(app_state) -> None:
    lock = getattr(app_state, "health_reinit_lock", None)
    if lock is None:
        lock = threading.Lock()
        app_state.health_reinit_lock = lock

    with lock:
        config = ConfigLoader.load("config/api_config.json")
        device = app_state.device
        if device is None:
            raise RuntimeError("Device not initialized")

        model, cfg = SealEmbeddingNet.from_package(
            pkg_dir=config.embedding_model.pkg_dir,
            device=device,
            verbose=False,
        )

        app_state.model = model
        app_state.config = config
        app_state.embedding_service = EmbeddingService(
            model,
            cfg,
            device,
            batch_size=config.batch_processing.embedding_batch_size,
        )
        app_state.detection_service = DetectionService(config.detection_model)
        
        milvus_db_path = config.milvus.db_path
        if not os.path.isabs(milvus_db_path):
            milvus_db_path = os.path.join(os.getcwd(), milvus_db_path)
        
        app_state.milvus_service = MilvusService(
            db_path=milvus_db_path,
            collection_name=config.milvus.collection_name,
        )
        
        if hasattr(app_state, "health_verify_cache"):
            delattr(app_state, "health_verify_cache")


async def _run_verify(
    detection_service,
    embedding_service,
    query_dir: Path,
    candidate_dir: Path,
    max_images: int,
    random_sample: bool,
):
    query_files = _sample_files(_list_images(query_dir), max_images, random_sample)
    candidate_files = _sample_files(_list_images(candidate_dir), max_images, random_sample)

    if not query_files or not candidate_files:
        raise HTTPException(status_code=400, detail="Verification data not found")

    query_images = [Image.open(p).convert("RGB") for p in query_files]
    candidate_images = [Image.open(p).convert("RGB") for p in candidate_files]

    query_seals = await detection_service.detect_seals(query_images)
    candidate_seals = await detection_service.detect_seals(candidate_images)

    if not query_seals or not candidate_seals:
        raise HTTPException(status_code=400, detail="No seals detected for verification")

    q_emb = await embedding_service.extract_embeddings_batch(query_seals)
    c_emb = await embedding_service.extract_embeddings_batch(candidate_seals)

    if q_emb.size == 0 or c_emb.size == 0:
        raise HTTPException(status_code=500, detail="Failed to extract embeddings")

    sims = q_emb @ c_emb.T
    top1_scores = np.max(sims, axis=1)

    return {
        "query_count": len(query_files),
        "candidate_count": len(candidate_files),
        "query_detected": len(query_seals),
        "candidate_detected": len(candidate_seals),
        "top1_avg": float(np.mean(top1_scores)),
        "top1_min": float(np.min(top1_scores)),
        "top1_max": float(np.max(top1_scores)),
    }


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    fastapi_request: Request,
    verify: bool = False,
    max_images: int = 3,
    random_sample: bool = True,
    force: bool = False,
    verify_ttl_sec: int = 300,
):
    async def _run_once():
        app_state = fastapi_request.app.state
        embedding_service = app_state.embedding_service
        detection_service = app_state.detection_service
        milvus_service = app_state.milvus_service

        if embedding_service is None or detection_service is None:
            raise RuntimeError("Services not initialized")

        milvus_loaded = False
        milvus_collection = "unknown"
        milvus_count = 0
        
        if milvus_service:
            try:
                milvus_loaded = True
                milvus_collection = milvus_service.collection_name
                milvus_count = milvus_service.count()
            except Exception as e:
                milvus_loaded = False
                milvus_collection = "error"

        response = HealthCheckResponse(
            status="healthy",
            model_loaded=True,
            embedding_dim=768,
            milvus_loaded=milvus_loaded,
            milvus_collection=milvus_collection,
            milvus_count=milvus_count,
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
            detection_service=detection_service,
            embedding_service=embedding_service,
            query_dir=Path("data/query"),
            candidate_dir=Path("data/candidate"),
            max_images=max_images,
            random_sample=random_sample,
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

    try:
        return await _run_once()
    except Exception:
        try:
            _reinit_services(fastapi_request.app.state)
            return await _run_once()
        except Exception as e:
            return HealthCheckResponse(
                status="unhealthy",
                model_loaded=False,
                embedding_dim=768,
                milvus_loaded=False,
                milvus_collection="unknown",
                milvus_count=0,
                message=str(e),
            )
