"""
Настройки приложения с использованием Pydantic
"""
import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения"""

    # Ollama настройки
    ollama_url: str = Field(default="http://localhost:11434", env="OLLAMA_URL")
    ollama_model: str = Field(
        default="yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest",
        env="OLLAMA_MODEL"
    )
    ollama_embedding_model: str = Field(default="nomic-embed-text", env="OLLAMA_EMBEDDING_MODEL")
    ollama_max_retries: int = Field(default=1, env="OLLAMA_MAX_RETRIES")
    ollama_retry_backoff: List[int] = Field(default=[0, 2, 5], env="OLLAMA_RETRY_BACKOFF")

    # Пути к базам данных
    db_profiles: str = Field(default="user_profiles.db", env="DB_PROFILES")
    db_embeddings: str = Field(default="./chroma_db", env="DB_EMBEDDINGS")

    # Версии форматов данных
    profiles_version: str = "1.0"
    embeddings_version: str = "1.0"

    # Telegram настройки
    telegram_bot_token: Optional[str] = Field(default=None, env="TELEGRAM_BOT_TOKEN")
    admin_user_id: Optional[int] = Field(default=None, env="ADMIN_USER_ID")
    allowed_chat_ids: List[int] = Field(default_factory=list, env="ALLOWED_CHAT_IDS")

    # Системные настройки
    log_dir: str = Field(default="logs", env="LOG_DIR")
    max_queue_size: int = 5

    # Файлы данных
    messages_file: str = "result.json"

    class Config:
        """Конфигурация Pydantic"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Глобальный экземпляр настроек
settings = Settings()