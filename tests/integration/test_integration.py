"""Integration flow test (runs only when Milvus db file exists)."""

import base64
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from seal_embedding_api.main import init_app


def _load_db_path() -> Path:
    """Read Milvus db path from config; skip if missing."""
    cfg_path = Path("config/api_config.json")
    if not cfg_path.exists():
        pytest.skip("Missing config/api_config.json")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    db_path = cfg.get("milvus", {}).get("db_path")
    if not db_path:
        pytest.skip("Missing milvus.db_path in config")
    return Path(db_path)


def _load_sample_image() -> Image.Image:
    """Load a real sample image from data/query or data/candidate."""
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
    for folder in (Path("data/query"), Path("data/candidate")):
        if not folder.exists():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in exts:
                return Image.open(p).convert("RGB")
    pytest.skip("No sample images found in data/query or data/candidate")


def _b64_image() -> str:
    """Create a base64-encoded image from real sample data."""
    img = _load_sample_image()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.mark.integration
def test_integration_flow():
    """Run ingest -> search -> delete against real Milvus Lite."""
    db_path = _load_db_path()
    if not db_path.exists():
        pytest.skip(f"Milvus db not found: {db_path}")

    app = init_app()
    with TestClient(app) as client:
        ingest_payload = {
            "items": [
                {"image_b64": _b64_image(), "seal_id": "it_uid_1"},
            ]
        }
        resp = client.post("/seals/ingest_base64", json=ingest_payload)
        assert resp.status_code == 200

        resp = client.post("/seals/search", json={"query_seal_id": "it_uid_1", "top_k": 1})
        assert resp.status_code == 200

        resp = client.post("/seals/delete", json={"seal_ids": ["it_uid_1"]})
        assert resp.status_code == 200
