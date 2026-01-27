"""
配置加载模块
- 读取 config/api_config.json
- 参数验证和默认值处理
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass

from seal_embedding_api.logger_config import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingModelConfig:
    """嵌入模型配置"""
    pkg_dir: str
    device: str = "cpu"


@dataclass
class DetectionModelConfig:
    """检测模型配置"""
    model_name: str
    batch_size: int = 1


@dataclass
class BatchProcessingConfig:
    """批处理配置"""
    embedding_batch_size: int = 32
    max_concurrent_tasks: int = 4


@dataclass
class StorageConfig:
    """存储配置"""
    base_dir: str


@dataclass
class ConfidenceThresholds:
    """置信度阈值"""
    high: float = 0.9
    medium: float = 0.7


@dataclass
class SimilaritySearchConfig:
    """相似度搜索配置"""
    default_top_k: int = 3


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    dir: str = "logs"


@dataclass
class ServerConfig:
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"


@dataclass
class APIConfig:
    """完整的 API 配置"""
    embedding_model: EmbeddingModelConfig
    detection_model: DetectionModelConfig
    batch_processing: BatchProcessingConfig
    storage: StorageConfig
    similarity_search: SimilaritySearchConfig
    logging: LoggingConfig
    server: ServerConfig


class ConfigLoader:
    """配置加载器"""

    @staticmethod
    def load(config_path: str = "config/api_config.json") -> APIConfig:
        """
        从 JSON 文件加载配置

        参数:
            config_path: 配置文件路径

        返回:
            APIConfig 对象

        异常:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置文件格式不正确或参数验证失败
        """
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

        # 验证必需的配置项
        ConfigLoader._validate_config(config_dict)

        # 构建配置对象
        api_config = ConfigLoader._build_config(config_dict)

        logger.info(f"配置加载成功")
        logger.debug(f"Embedding model: {api_config.embedding_model.pkg_dir}")
        logger.debug(f"Detection model: {api_config.detection_model.model_name}")
        logger.debug(f"Batch size: {api_config.batch_processing.embedding_batch_size}")

        return api_config

    @staticmethod
    def _validate_config(config_dict: Dict[str, Any]) -> None:
        """验证配置项"""
        required_keys = [
            "embedding_model",
            "detection_model",
            "batch_processing",
            "storage",
        ]

        for key in required_keys:
            if key not in config_dict:
                raise ValueError(f"Missing required config key: {key}")

        # 验证 batch_size 范围
        batch_size = config_dict.get("batch_processing", {}).get("embedding_batch_size", 32)
        if not (1 <= batch_size <= 256):
            raise ValueError(
                f"embedding_batch_size must be between 1 and 256, got {batch_size}"
            )

        # 验证 device
        device = config_dict.get("embedding_model", {}).get("device", "cpu")
        if device not in ["cpu", "cuda"]:
            raise ValueError(f"device must be 'cpu' or 'cuda', got {device}")

    @staticmethod
    def _build_config(config_dict: Dict[str, Any]) -> APIConfig:
        """从字典构建配置对象"""
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
            max_concurrent_tasks=config_dict["batch_processing"].get("max_concurrent_tasks", 4),
        )

        storage = StorageConfig(
            base_dir=config_dict["storage"]["base_dir"],
        )

        similarity_search_dict = config_dict.get("similarity_search", {})
        similarity_search = SimilaritySearchConfig(
            default_top_k=similarity_search_dict.get("default_top_k", 3),
        )

        logging_config = LoggingConfig(
            level=config_dict.get("logging", {}).get("level", "INFO"),
            dir=config_dict.get("logging", {}).get("dir", "logs"),
        )

        server_config = config_dict.get("server", {})
        server = ServerConfig(
            host=server_config.get("host", "0.0.0.0"),
            port=server_config.get("port", 8000),
            reload=server_config.get("reload", False),
            log_level=server_config.get("log_level", "info"),
        )

        return APIConfig(
            embedding_model=embedding_model,
            detection_model=detection_model,
            batch_processing=batch_processing,
            storage=storage,
            similarity_search=similarity_search,
            logging=logging_config,
            server=server,
        )
