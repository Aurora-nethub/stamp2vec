import logging
import logging.handlers
from pathlib import Path
from typing import Optional


class LoggerConfig:
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
        logger = logging.getLogger(name)

        if logger.hasHandlers():
            return logger

        logger.setLevel(getattr(logging, log_level.upper()))

        formatter = logging.Formatter(log_format, datefmt=date_format)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        log_file = log_dir_path / f"seal_api.log"

        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file),
            maxBytes=100 * 1024 * 1024,
            backupCount=30,
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
        if log_level is None:
            log_level = LoggerConfig.DEFAULT_LOG_LEVEL

        return LoggerConfig.setup_logger(
            name=name,
            log_level=log_level,
            log_dir=log_dir,
        )


def get_logger(name: str, log_level: Optional[str] = None) -> logging.Logger:
    return LoggerConfig.get_logger(name, log_level)
