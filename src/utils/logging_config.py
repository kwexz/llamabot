"""
Настройка логирования приложения
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from config.settings import settings


class TimestampedFormatter(logging.Formatter):
    """Форматтер с timestamp для логов"""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = super().format(record)
        return f"{timestamp} {message}"


def setup_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: Optional[str] = None
) -> None:
    """
    Настраивает логирование приложения

    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Включить логирование в файл
        log_dir: Директория для логов (если None, используется из настроек)
    """
    if log_dir is None:
        log_dir = settings.log_dir

    # Создаем директорию для логов
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Настраиваем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Очищаем существующие обработчики
    root_logger.handlers.clear()

    # Форматтер для логов
    formatter = TimestampedFormatter(
        '%(levelname)s [%(name)s]: %(message)s'
    )

    # Консольный обработчик
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Файловый обработчик
    if log_to_file:
        log_file = log_path / f"{datetime.now():%Y-%m-%d}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Получить логгер с указанным именем

    Args:
        name: Имя логгера

    Returns:
        Логгер
    """
    return logging.getLogger(name)