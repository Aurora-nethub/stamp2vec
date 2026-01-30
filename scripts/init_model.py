#!/usr/bin/env python
import json
import os
import sys
from pathlib import Path

from paddleocr import LayoutDetection

from seal_embedding_api.logger_config import get_logger

logger = get_logger(__name__)


ROOT_DIR = Path(__file__).parent.parent


def load_model_name(config_path: str = "config/api_config.json") -> str:
    cfg_file = ROOT_DIR / config_path
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")
    with cfg_file.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_name = cfg.get("detection_model", {}).get("model_name")
    if not model_name:
        raise KeyError("detection_model.model_name is missing in config")
    return model_name


def model_cache_dir(model_name: str) -> Path:
    home = Path(os.path.expanduser("~"))
    return home / ".paddlex" / "official_models" / model_name


def init_model():
    model_name = load_model_name()
    cache_dir = model_cache_dir(model_name)
    if cache_dir.exists() and any(cache_dir.iterdir()):
        logger.info(f"Model already cached: {cache_dir}")
        return True

    logger.info(f"Downloading PaddleOCR model: {model_name}")
    LayoutDetection(model_name=model_name)

    if cache_dir.exists() and any(cache_dir.iterdir()):
        logger.info(f"Model download complete: {cache_dir}")
        return True

    logger.error(f"Model download failed or cache not found: {cache_dir}")
    return False


if __name__ == "__main__":
    try:
        ok = init_model()
        sys.exit(0 if ok else 1)
    except Exception as e:
        logger.error(f"Init model failed: {e}", exc_info=True)
        sys.exit(1)
