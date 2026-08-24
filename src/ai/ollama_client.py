"""
Клиент для взаимодействия с Ollama API
"""
import asyncio
import time
from typing import List, Optional
import requests
import numpy as np

from config.settings import settings
from utils import get_logger

logger = get_logger(__name__)


class OllamaClient:
    """Клиент для взаимодействия с Ollama API"""

    def __init__(self):
        self.base_url = settings.ollama_url.rstrip('/')
        self.max_retries = settings.ollama_max_retries
        self.retry_backoff = settings.ollama_retry_backoff

    def _make_request(self, endpoint: str, data: dict, timeout: int = 300) -> dict:
        """
        Выполняет HTTP запрос к Ollama API с повторными попытками

        Args:
            endpoint: Конечная точка API
            data: Данные для отправки
            timeout: Таймаут запроса

        Returns:
            Ответ от API

        Raises:
            Exception: При всех неудачных попытках
        """
        url = f"{self.base_url}{endpoint}"

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(url, json=data, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                error_msg = f"Ошибка при обращении к Ollama {endpoint} (попытка {attempt}/{self.max_retries}): {e}"
                logger.warning(error_msg)

                if attempt == self.max_retries:
                    raise Exception(error_msg)

                # Задержка перед следующей попыткой
                backoff = self.retry_backoff[min(attempt, len(self.retry_backoff) - 1)]
                if backoff > 0:
                    time.sleep(backoff)

    def generate_text(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Генерирует текст с помощью Ollama

        Args:
            prompt: Промпт для генерации
            model: Модель для использования (если None, используется настройка по умолчанию)

        Returns:
            Сгенерированный текст
        """
        if model is None:
            model = settings.ollama_model

        data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = self._make_request("/api/generate", data)
            return response.get("response", "").strip()
        except Exception as e:
            logger.error(f"Не удалось сгенерировать текст: {e}")
            return f"Ошибка генерации текста: {e}"

    def get_embedding(self, text: str, model: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Получает эмбеддинг для текста

        Args:
            text: Текст для эмбеддинга
            model: Модель для эмбеддингов (если None, используется настройка по умолчанию)

        Returns:
            Вектор эмбеддинга или None при ошибке
        """
        if model is None:
            model = settings.ollama_embedding_model

        data = {
            "model": model,
            "prompt": text
        }

        try:
            response = self._make_request("/api/embeddings", data, timeout=60)
            embedding = response.get("embedding")
            if embedding:
                return np.array(embedding, dtype=np.float32)
            return None
        except Exception as e:
            logger.error(f"Не удалось получить эмбеддинг: {e}")
            return None

    def get_available_models(self) -> List[str]:
        """
        Получает список доступных моделей

        Returns:
            Список имен моделей
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=30)
            response.raise_for_status()
            result = response.json()
            models = result.get("models", [])
            return [model.get("name", "") for model in models if model.get("name")]
        except Exception as e:
            logger.error(f"Не удалось получить список моделей: {e}")
            return []


class AsyncOllamaClient:
    """Асинхронная версия клиента Ollama"""

    def __init__(self):
        self.client = OllamaClient()

    async def generate_text(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Асинхронно генерирует текст
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.client.generate_text, prompt, model)

    async def get_embedding(self, text: str, model: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Асинхронно получает эмбеддинг
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.client.get_embedding, text, model)

    async def get_available_models(self) -> List[str]:
        """
        Асинхронно получает список моделей
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.client.get_available_models)