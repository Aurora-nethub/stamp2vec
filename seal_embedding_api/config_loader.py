import json
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass

from seal_embedding_api.logger_config import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingModelConfig:
    pkg_dir: str
    device: str = "cpu"


@dataclass
class DetectionModelConfig:
    model_name: str
    batch_size: int = 1


@dataclass
class BatchProcessingConfig:
    embedding_batch_size: int = 32


@dataclass
class MilvusConfig:
    db_path: str
    collection_name: str = "seals"
    dimension: int = 768


@dataclass
class APIConfig:
    embedding_model: EmbeddingModelConfig
    detection_model: DetectionModelConfig
    batch_processing: BatchProcessingConfig
    milvus: MilvusConfig


class ConfigLoader:
    @staticmethod
    def load(config_path: str = "config/api_config.json") -> APIConfig:
        config_file = Path(config_path)

        if not config_file.exists():
            logger.error(f"配置文件不存在: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logger.info(f"加载配置文件: {config_path}")

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"配置文件 JSON 格式错误: {e}")
            raise ValueError(f"Invalid JSON in config file: {e}")

        ConfigLoader._validate_config(config_dict)

        api_config = ConfigLoader._build_config(config_dict)

        logger.info(f"配置加载成功")
        logger.debug(f"Embedding model: {api_config.embedding_model.pkg_dir}")
        logger.debug(f"Detection model: {api_config.detection_model.model_name}")
        logger.debug(f"Batch size: {api_config.batch_processing.embedding_batch_size}")

        return api_config

    @staticmethod
    def _validate_config(config_dict: Dict[str, Any]) -> None:
        required_keys = [
            "embedding_model",
            "detection_model",
            "batch_processing",
        ]

        for key in required_keys:
            if key not in config_dict:
                raise ValueError(f"Missing required config key: {key}")

        batch_size = config_dict.get("batch_processing", {}).get("embedding_batch_size", 32)
        if not (1 <= batch_size <= 256):
            raise ValueError(
                f"embedding_batch_size must be between 1 and 256, got {batch_size}"
            )

        device = config_dict.get("embedding_model", {}).get("device", "cpu")
        if device not in ["cpu", "cuda"]:
            raise ValueError(f"device must be 'cpu' or 'cuda', got {device}")

    @staticmethod
    def _build_config(config_dict: Dict[str, Any]) -> APIConfig:
        embedding_model = EmbeddingModelConfig(
            pkg_dir=config_dict["embedding_model"]["pkg_dir"],
            device=config_dict["embedding_model"].get("device", "cpu"),
        )

        detection_model = DetectionModelConfig(
            model_name=config_dict["detection_model"]["model_name"],
            batch_size=config_dict["detection_model"].get("batch_size", 1),
        )

        batch_processing = BatchProcessingConfig(
            embedding_batch_size=config_dict["batch_processing"].get("embedding_batch_size", 32),
        )

        milvus_dict = config_dict.get("milvus", {})
        milvus = MilvusConfig(
            db_path=milvus_dict.get("db_path", "database/milvus"),
            collection_name=milvus_dict.get("collection_name", "seals"),
            dimension=milvus_dict.get("dimension", 768),
        )

        return APIConfig(
            embedding_model=embedding_model,
            detection_model=detection_model,
            batch_processing=batch_processing,
            milvus=milvus,
        )
