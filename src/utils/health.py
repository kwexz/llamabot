"""
Модуль мониторинга здоровья и метрик приложения
"""
import time
import psutil
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from config.settings import settings
from ai.ollama_client import OllamaClient
from database.repository import UserRepository, SettingsRepository
from utils import get_logger

logger = get_logger(__name__)


class HealthChecker:
    """Проверка здоровья компонентов системы"""

    def __init__(self):
        self.user_repo = UserRepository()
        self.settings_repo = SettingsRepository()
        self.ollama_client = OllamaClient()
        self.start_time = time.time()
        self.last_health_check = None

    def check_database(self) -> Dict[str, Any]:
        """
        Проверка здоровья базы данных

        Returns:
            Статус проверки БД
        """
        try:
            # Проверяем подключение
            settings = self.settings_repo.get_all_settings()

            # Проверяем наличие основных данных
            profiles_count = len(self.user_repo.get_all_profiles())

            return {
                "status": "healthy",
                "database_type": "SQLite",
                "profiles_count": profiles_count,
                "settings_count": len(settings),
                "last_check": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Ошибка проверки базы данных: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }

    def check_ollama(self) -> Dict[str, Any]:
        """
        Проверка здоровья Ollama сервиса

        Returns:
            Статус проверки Ollama
        """
        try:
            # Проверяем доступность моделей
            models = self.ollama_client.get_available_models()

            # Проверяем генерацию текста
            test_response = self.ollama_client.generate_text("Test message", timeout=10)

            return {
                "status": "healthy",
                "models_available": len(models),
                "current_model": settings.ollama_model,
                "embedding_model": settings.ollama_embedding_model,
                "test_generation": bool(test_response),
                "last_check": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Ошибка проверки Ollama: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }

    def check_embeddings(self) -> Dict[str, Any]:
        """
        Проверка здоровья системы эмбеддингов

        Returns:
            Статус проверки эмбеддингов
        """
        try:
            from embeddings.manager import EmbeddingManager

            manager = EmbeddingManager()
            stats = manager.get_embeddings_stats()

            total_embeddings = sum(
                user_stats.get('embeddings_count', 0)
                for user_stats in stats.values()
            )

            return {
                "status": "healthy",
                "total_embeddings": total_embeddings,
                "users_with_embeddings": len(stats),
                "chromadb_status": "connected" if manager.collection else "disconnected",
                "last_check": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Ошибка проверки эмбеддингов: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }

    def check_system_resources(self) -> Dict[str, Any]:
        """
        Проверка системных ресурсов

        Returns:
            Информация о ресурсах системы
        """
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu = psutil.cpu_percent(interval=1)

            return {
                "status": "healthy",
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "percent_used": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent_used": disk.percent
                },
                "cpu_percent": cpu,
                "uptime_seconds": time.time() - self.start_time,
                "last_check": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Ошибка проверки системных ресурсов: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }

    def get_full_health_status(self) -> Dict[str, Any]:
        """
        Полная проверка здоровья системы

        Returns:
            Полный статус здоровья всех компонентов
        """
        self.last_health_check = datetime.now()

        health_status = {
            "timestamp": self.last_health_check.isoformat(),
            "service": "telegram-bot",
            "version": settings.profiles_version,
            "checks": {
                "database": self.check_database(),
                "ollama": self.check_ollama(),
                "embeddings": self.check_embeddings(),
                "system": self.check_system_resources()
            }
        }

        # Определяем общий статус
        all_healthy = all(
            check.get("status") == "healthy"
            for check in health_status["checks"].values()
        )

        health_status["status"] = "healthy" if all_healthy else "degraded"

        logger.info(f"Health check completed: {health_status['status']}")
        return health_status

    def is_healthy(self) -> bool:
        """
        Быстрая проверка общего здоровья

        Returns:
            True если система здорова
        """
        try:
            # Проверяем основные компоненты
            db_status = self.check_database()
            ollama_status = self.check_ollama()

            return (
                db_status.get("status") == "healthy" and
                ollama_status.get("status") == "healthy"
            )
        except Exception:
            return False


class MetricsCollector:
    """Сборщик метрик приложения"""

    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "ai_requests_total": 0,
            "ai_requests_success": 0,
            "ai_requests_failed": 0,
            "embeddings_created": 0,
            "messages_processed": 0,
            "uptime_seconds": 0
        }
        self.start_time = time.time()

    def increment(self, metric: str, value: int = 1) -> None:
        """
        Увеличивает значение метрики

        Args:
            metric: Название метрики
            value: Значение для добавления
        """
        if metric in self.metrics:
            self.metrics[metric] += value
        else:
            logger.warning(f"Неизвестная метрика: {metric}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Получает все метрики

        Returns:
            Словарь с метриками
        """
        metrics = self.metrics.copy()
        metrics["uptime_seconds"] = time.time() - self.start_time
        metrics["timestamp"] = datetime.now().isoformat()

        # Вычисляем производные метрики
        total_requests = metrics["requests_total"]
        if total_requests > 0:
            metrics["success_rate"] = metrics["requests_success"] / total_requests
        else:
            metrics["success_rate"] = 0.0

        total_ai = metrics["ai_requests_total"]
        if total_ai > 0:
            metrics["ai_success_rate"] = metrics["ai_requests_success"] / total_ai
        else:
            metrics["ai_success_rate"] = 0.0

        return metrics

    def reset(self) -> None:
        """Сбрасывает все метрики"""
        for key in self.metrics:
            if key != "uptime_seconds":  # Не сбрасываем uptime
                self.metrics[key] = 0

        logger.info("Метрики сброшены")


# Глобальные экземпляры для использования в приложении
health_checker = HealthChecker()
metrics_collector = MetricsCollector()


def get_health_status() -> Dict[str, Any]:
    """
    Получить полный статус здоровья системы

    Returns:
        Статус здоровья
    """
    return health_checker.get_full_health_status()


def record_request(success: bool = True) -> None:
    """
    Записать запрос в метрики

    Args:
        success: Успешность запроса
    """
    metrics_collector.increment("requests_total")
    if success:
        metrics_collector.increment("requests_success")
    else:
        metrics_collector.increment("requests_failed")


def record_ai_request(success: bool = True) -> None:
    """
    Записать AI запрос в метрики

    Args:
        success: Успешность запроса
    """
    metrics_collector.increment("ai_requests_total")
    if success:
        metrics_collector.increment("ai_requests_success")
    else:
        metrics_collector.increment("ai_requests_failed")


def record_embedding_created() -> None:
    """Записать создание эмбеддинга"""
    metrics_collector.increment("embeddings_created")


def record_message_processed() -> None:
    """Записать обработку сообщения"""
    metrics_collector.increment("messages_processed")