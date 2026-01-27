"""
日志配置模块
- 初始化日志系统（Console + File Handler）
- 为各个模块提供已配置的 logger
- 支持日志轮转和保留策略
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


class LoggerConfig:
    """日志配置类"""

    DEFAULT_LOG_DIR = "logs"
    DEFAULT_LOG_LEVEL = "INFO"
    DEFAULT_LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    @staticmethod
    def setup_logger(
        name: str,
        log_level: str = DEFAULT_LOG_LEVEL,
        log_dir: str = DEFAULT_LOG_DIR,
        log_format: str = DEFAULT_LOG_FORMAT,
        date_format: str = DEFAULT_DATE_FORMAT,
    ) -> logging.Logger:
        """
        为指定的模块创建和配置 logger

        参数:
            name: logger 名称（通常是 __name__）
            log_level: 日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）
            log_dir: 日志文件目录
            log_format: 日志格式字符串
            date_format: 日期格式字符串

        返回:
            配置好的 logger 对象
        """
        logger = logging.getLogger(name)

        # 避免重复添加 handler
        if logger.hasHandlers():
            return logger

        # 设置日志级别
        logger.setLevel(getattr(logging, log_level.upper()))

        # 创建格式化器
        formatter = logging.Formatter(log_format, datefmt=date_format)

        # 1. 控制台 Handler（开发时实时输出）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. 文件 Handler（持久化）- 使用 RotatingFileHandler
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        log_file = log_dir_path / f"seal_api.log"

        # RotatingFileHandler: 按大小和备份数轮转
        # 100MB 时轮转，最多保留 30 个备份（≈3GB）
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file),
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=30,  # 保留最近 30 个备份
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    @staticmethod
    def get_logger(
        name: str,
        log_level: Optional[str] = None,
        log_dir: str = DEFAULT_LOG_DIR,
    ) -> logging.Logger:
        """
        获取指定名称的 logger（快捷方法）

        参数:
            name: logger 名称（通常是 __name__）
            log_level: 日志级别（如果为 None，使用默认值）
            log_dir: 日志文件目录

        返回:
            配置好的 logger 对象
        """
        if log_level is None:
            log_level = LoggerConfig.DEFAULT_LOG_LEVEL

        return LoggerConfig.setup_logger(
            name=name,
            log_level=log_level,
            log_dir=log_dir,
        )


# 方便使用：全局函数
def get_logger(name: str, log_level: Optional[str] = None) -> logging.Logger:
    """快速获取 logger 的全局函数"""
    return LoggerConfig.get_logger(name, log_level)
