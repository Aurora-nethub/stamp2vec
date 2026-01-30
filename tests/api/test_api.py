"""API tests using stubbed services (no real model or Milvus)."""

import base64
import io

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from seal_embedding_api.api import health, pipeline


class FakeDetectionService:
    """Returns input images directly as detected seals."""
    async def detect_seals(self, images):
        return images


class FakeEmbeddingService:
    """Returns dummy embeddings for predictable API behavior."""
    async def extract_embeddings_batch(self, seals):
        if not seals:
            return np.array([]).reshape(0, 768)
        return np.ones((len(seals), 768), dtype=np.float32)

    async def extract_embedding(self, seal):
        return np.ones((768,), dtype=np.float32)


class FakeMilvusService:
    """In-memory store that mimics Milvus operations."""
    def __init__(self):
        self._store = {}
        self.collection_name = "seals"

    def insert_embedding(self, seal_id, embedding):
        self._store[seal_id] = embedding
        return True

    def search_similar(self, query_embedding, top_k=3, exclude_id=None):
        results = []
        for k in self._store.keys():
            if exclude_id and k == exclude_id:
                continue
            results.append({"id": k, "similarity": 0.9})
        return results[:top_k]

    def get_embedding(self, seal_id):
        return self._store.get(seal_id)

    def delete(self, seal_ids):
        succeeded = []
        failed = []
        for sid in seal_ids:
            if sid in self._store:
                del self._store[sid]
                succeeded.append(sid)
            else:
                failed.append(sid)
        return succeeded, failed

    def count(self):
        return len(self._store)


def _make_app():
    """Construct a FastAPI app with fake dependencies."""
    app = FastAPI()
    app.include_router(health.router)
    app.include_router(pipeline.router)

    app.state.detection_service = FakeDetectionService()
    app.state.embedding_service = FakeEmbeddingService()
    app.state.milvus_service = FakeMilvusService()

    class _Cfg:
        class _Det:
            batch_size = 1
        class _Batch:
            embedding_batch_size = 4
        detection_model = _Det()
        batch_processing = _Batch()

    app.state.config = _Cfg()
    return app


@pytest.fixture()
def client():
    """Test client bound to the fake app."""
    app = _make_app()
    return TestClient(app)


def _b64_image():
    """Create a small base64-encoded PNG image."""
    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_health_basic(client):
    """Health endpoint returns a valid status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] in {"healthy", "unhealthy"}


def test_ingest_base64_success(client):
    """Ingest base64 images and store embeddings."""
    payload = {
        "items": [
            {"image_b64": _b64_image(), "seal_id": "uid_1"},
            {"image_b64": _b64_image(), "seal_id": "uid_2"},
        ]
    }
    resp = client.post("/seals/ingest_base64", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["seals_detected"] == 2
    assert len(data["embeddings_stored"]) == 2


def test_search_by_id_not_found(client):
    """Search by missing seal_id returns 404."""
    resp = client.post("/seals/search", json={"query_seal_id": "missing", "top_k": 1})
    assert resp.status_code == 404


def test_delete_empty(client):
    """Delete with empty seal_ids returns 400."""
    resp = client.post("/seals/delete", json={"seal_ids": []})
    assert resp.status_code == 400
