"""
Менеджер эмбеддингов для работы с ChromaDB
"""
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from database.repository import MessageEmbeddingRepository
from ai.ollama_client import OllamaClient
from utils import get_logger

logger = get_logger(__name__)


class EmbeddingManager:
    """Менеджер для работы с эмбеддингами сообщений в ChromaDB"""

    def __init__(self):
        self.chroma_client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[chromadb.Collection] = None
        self.repository = MessageEmbeddingRepository()
        self.ollama_client = OllamaClient()
        self._initialize_chromadb()

    def _initialize_chromadb(self) -> None:
        """Инициализирует ChromaDB"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=settings.db_embeddings,
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            # Создаем или получаем коллекцию
            try:
                self.collection = self.chroma_client.get_collection("message_embeddings")
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    "message_embeddings",
                    metadata={"hnsw:space": "cosine"}
                )

            logger.info("✓ ChromaDB инициализирована успешно")
            self._log_collection_stats()

        except Exception as e:
            logger.error(f"Ошибка при инициализации ChromaDB: {e}")
            self.chroma_client = None
            self.collection = None

    def _log_collection_stats(self) -> None:
        """Логирует статистику коллекции"""
        if self.collection:
            try:
                total_embeddings = len(self.collection.get()['ids'])
                logger.info(f"ChromaDB collection 'message_embeddings' содержит {total_embeddings} эмбеддингов")
            except Exception as e:
                logger.warning(f"Не удалось получить статистику коллекции: {e}")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Вычисляет косинусное сходство между двумя векторами

        Args:
            vec1: Первый вектор
            vec2: Второй вектор

        Returns:
            Косинусное сходство (от -1 до 1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def create_embeddings_for_user(
        self,
        user_id: int,
        canonical_name: str,
        messages: List[str],
        force_recreate: bool = False
    ) -> int:
        """
        Создает эмбеддинги для всех сообщений пользователя

        Args:
            user_id: ID пользователя
            canonical_name: Каноническое имя пользователя
            messages: Список сообщений
            force_recreate: Принудительно пересоздать все эмбеддинги

        Returns:
            Количество созданных эмбеддингов
        """
        if not messages or not self.collection:
            return 0

        logger.info(f"Создание эмбеддингов для {canonical_name} ({user_id}) - {len(messages)} сообщений")

        # Подсчет существующих эмбеддингов
        existing_count = 0
        if not force_recreate:
            try:
                results = self.collection.get(where={"user_id": user_id})
                existing_count = len(results['ids'])
            except Exception as e:
                logger.debug(f"Ошибка при подсчете существующих эмбеддингов: {e}")
                existing_count = 0

        logger.debug(f"Для {canonical_name} уже существует {existing_count} эмбеддингов из {len(messages)}")

        if existing_count >= len(messages) and not force_recreate:
            logger.info(f"Эмбеддинги для {canonical_name} уже существуют, пропускаем")
            return 0

        created_count = 0

        for i, message in enumerate(messages):
            if (i + 1) % 50 == 0:
                logger.info(f"Обработано {i + 1}/{len(messages)} сообщений...")

            # Проверяем, не обработано ли сообщение уже
            if not force_recreate and self._is_message_processed(user_id, message):
                continue

            # Получаем эмбеддинг
            embedding = self.ollama_client.get_embedding(message)
            if embedding is None:
                logger.warning(f"Не удалось получить эмбеддинг для сообщения пользователя {user_id}")
                continue

            # Генерируем уникальный ID
            import hashlib
            doc_id = hashlib.md5(f"{user_id}_{message}_{np.random.rand()}".encode()).hexdigest()

            try:
                # Сохраняем в ChromaDB
                self.collection.add(
                    documents=[message],
                    embeddings=[embedding.tolist()],
                    ids=[doc_id],
                    metadatas=[{
                        "user_id": user_id,
                        "canonical_name": canonical_name,
                        "message_text": message,
                        "timestamp": np.datetime64('now').astype(str)
                    }]
                )

                # Сохраняем в SQLite для резервной копии
                self.repository.save_embedding(canonical_name, message, embedding.tolist(), "chromadb")

                created_count += 1

            except Exception as e:
                logger.error(f"Ошибка сохранения эмбеддинга для {canonical_name}: {e}")
                continue

        logger.info(f"✓ Создано {created_count} эмбеддингов для {canonical_name}")
        return created_count

    def _is_message_processed(self, user_id: int, message_text: str) -> bool:
        """
        Проверяет, обработано ли сообщение уже

        Args:
            user_id: ID пользователя
            message_text: Текст сообщения

        Returns:
            True если сообщение обработано
        """
        try:
            if self.collection:
                # Проверяем в ChromaDB
                results = self.collection.get(
                    where={"$and": [{"user_id": user_id}, {"message_text": message_text}]},
                    limit=1
                )
                return len(results['ids']) > 0
        except Exception as e:
            logger.debug(f"Ошибка проверки в ChromaDB: {e}")

        # Fallback к SQLite
        return self.repository.is_message_processed(str(user_id), message_text)

    def retrieve_relevant_messages(
        self,
        user_id: int,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Находит наиболее релевантные сообщения для запроса (RAG)

        Args:
            user_id: ID пользователя
            query: Запрос пользователя
            top_k: Количество результатов

        Returns:
            Список кортежей (сообщение, сходство)
        """
        if not self.collection:
            logger.warning("ChromaDB недоступна для поиска")
            return []

        try:
            # Получаем эмбеддинг запроса
            query_embedding = self.ollama_client.get_embedding(query)
            if query_embedding is None:
                logger.warning("Не удалось получить эмбеддинг для запроса")
                return []

            # Поиск в ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where={"user_id": user_id}
            )

            # Обрабатываем результаты
            relevant_messages = []
            if results['documents'] and results['distances']:
                for doc, distance in zip(results['documents'][0], results['distances'][0]):
                    # Для косинусного расстояния: чем меньше расстояние, тем больше сходство
                    similarity = 1 / (1 + distance) if distance > 0 else 1.0
                    relevant_messages.append((doc, similarity))

            logger.debug(f"Найдено {len(relevant_messages)} релевантных сообщений для user_id {user_id}")
            return relevant_messages

        except Exception as e:
            logger.error(f"Ошибка при поиске релевантных сообщений: {e}")
            return []

    def save_message_embedding(
        self,
        user_id: int,
        message_text: str,
        canonical_name: Optional[str] = None
    ) -> bool:
        """
        Сохраняет эмбеддинг для нового сообщения

        Args:
            user_id: ID пользователя
            message_text: Текст сообщения
            canonical_name: Каноническое имя пользователя

        Returns:
            True при успехе
        """
        if not self.collection:
            logger.warning("ChromaDB недоступна")
            return False

        try:
            # Проверяем, не обработано ли уже
            if self._is_message_processed(user_id, message_text):
                return True

            # Получаем эмбеддинг
            embedding = self.ollama_client.get_embedding(message_text)
            if embedding is None:
                logger.warning(f"Не удалось получить эмбеддинг для сообщения от user_id {user_id}")
                return False

            # Генерируем ID
            import hashlib
            doc_id = hashlib.md5(f"{user_id}_{message_text}_{np.random.rand()}".encode()).hexdigest()

            # Сохраняем в ChromaDB
            metadata = {
                "user_id": user_id,
                "message_text": message_text,
                "timestamp": np.datetime64('now').astype(str)
            }
            if canonical_name:
                metadata["canonical_name"] = canonical_name

            self.collection.add(
                documents=[message_text],
                embeddings=[embedding.tolist()],
                ids=[doc_id],
                metadatas=[metadata]
            )

            # Сохраняем в SQLite
            display_name = canonical_name or str(user_id)
            self.repository.save_embedding(display_name, message_text, embedding.tolist(), "telegram")

            logger.debug(f"✓ Сохранен эмбеддинг для нового сообщения от {display_name}")
            return True

        except Exception as e:
            display_name = canonical_name or str(user_id)
            logger.error(f"Ошибка сохранения эмбеддинга для {display_name}: {e}")
            return False

    def get_embeddings_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Получает статистику эмбеддингов для всех пользователей

        Returns:
            Статистика по пользователям
        """
        stats = {}

        if not self.collection:
            return stats

        try:
            # Получаем все метаданные
            all_data = self.collection.get()
            metadatas = all_data.get('metadatas', [])

            # Группируем по пользователям
            from collections import defaultdict
            user_stats = defaultdict(int)

            for metadata in metadatas:
                if metadata and 'user_id' in metadata:
                    user_id = metadata['user_id']
                    canonical_name = metadata.get('canonical_name', f'user_{user_id}')
                    user_stats[canonical_name] += 1

            # Преобразуем в итоговую структуру
            for canonical_name, count in user_stats.items():
                stats[canonical_name] = {
                    'embeddings_count': count,
                    'user_id': None  # TODO: получить из репозитория если нужно
                }

        except Exception as e:
            logger.error(f"Ошибка получения статистики эмбеддингов: {e}")

        return stats

    def clear_user_embeddings(self, canonical_name: str) -> bool:
        """
        Очищает эмбеддинги для пользователя

        Args:
            canonical_name: Каноническое имя пользователя

        Returns:
            True при успехе
        """
        if not self.collection:
            return False

        try:
            # Удаляем из ChromaDB
            self.collection.delete(where={"canonical_name": canonical_name})

            # TODO: Очистить из SQLite если нужно

            logger.info(f"✓ Очищены эмбеддинги для {canonical_name}")
            return True

        except Exception as e:
            logger.error(f"Ошибка очистки эмбеддингов для {canonical_name}: {e}")
            return False