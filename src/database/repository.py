"""
Репозиторий для работы с данными пользователей и профилей
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
import sqlite3

from database.connection import get_db_connection, ensure_message_embeddings_table
from database.models import UserProfile, Alias, Settings, Metadata
from config.settings import settings
from utils import get_logger

logger = get_logger(__name__)


class UserRepository:
    """Репозиторий для работы с данными пользователей"""

    def save_profile(self, profile: UserProfile) -> bool:
        """
        Сохраняет профиль пользователя

        Args:
            profile: Профиль пользователя

        Returns:
            True при успехе, False при ошибке
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO profiles
                    (user_id, canonical_name, profile_text, training_date, version, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    profile.user_id,
                    profile.canonical_name,
                    profile.profile_text,
                    profile.training_date,
                    profile.version
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Ошибка сохранения профиля {profile.canonical_name}: {e}")
            return False

    def get_profile(self, canonical_name: str) -> Optional[UserProfile]:
        """
        Получает профиль пользователя по каноническому имени

        Args:
            canonical_name: Каноническое имя пользователя

        Returns:
            Профиль пользователя или None
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT * FROM profiles WHERE canonical_name = ?',
                    (canonical_name,)
                )
                row = cursor.fetchone()
                if row:
                    return UserProfile(
                        user_id=row['user_id'],
                        canonical_name=row['canonical_name'],
                        profile_text=row['profile_text'],
                        training_date=row['training_date'],
                        version=row['version'],
                        updated_at=row['updated_at']
                    )
                return None
        except Exception as e:
            logger.error(f"Ошибка получения профиля {canonical_name}: {e}")
            return None

    def get_all_profiles(self) -> List[UserProfile]:
        """
        Получает все профили пользователей

        Returns:
            Список профилей
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM profiles ORDER BY canonical_name')
                rows = cursor.fetchall()
                return [
                    UserProfile(
                        user_id=row['user_id'],
                        canonical_name=row['canonical_name'],
                        profile_text=row['profile_text'],
                        training_date=row['training_date'],
                        version=row['version'],
                        updated_at=row['updated_at']
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Ошибка получения всех профилей: {e}")
            return []

    def save_aliases(self, user_id: int, aliases: List[str]) -> bool:
        """
        Сохраняет алиасы пользователя

        Args:
            user_id: ID пользователя
            aliases: Список алиасов

        Returns:
            True при успехе, False при ошибке
        """
        if not aliases:
            return True

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                for alias in aliases:
                    if alias.strip():
                        cursor.execute('''
                            INSERT OR IGNORE INTO aliases (user_id, alias)
                            VALUES (?, ?)
                        ''', (user_id, alias.strip()))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Ошибка сохранения алиасов для user_id {user_id}: {e}")
            return False

    def get_user_aliases(self, user_id: int) -> List[str]:
        """
        Получает все алиасы пользователя

        Args:
            user_id: ID пользователя

        Returns:
            Список алиасов
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT alias FROM aliases WHERE user_id = ?', (user_id,))
                rows = cursor.fetchall()
                return [row['alias'] for row in rows]
        except Exception as e:
            logger.error(f"Ошибка получения алиасов для user_id {user_id}: {e}")
            return []

    def find_user_by_alias(self, alias: str) -> Optional[str]:
        """
        Находит каноническое имя пользователя по алиасу

        Args:
            alias: Алиас пользователя

        Returns:
            Каноническое имя или None
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT canonical_name FROM profiles
                    JOIN aliases ON profiles.user_id = aliases.user_id
                    WHERE aliases.alias = ?
                ''', (alias.lower(),))
                row = cursor.fetchone()
                return row['canonical_name'] if row else None
        except Exception as e:
            logger.error(f"Ошибка поиска пользователя по алиасу '{alias}': {e}")
            return None

    def get_name_to_user_id_mapping(self) -> Dict[str, int]:
        """
        Получает маппинг канонических имен на user_id

        Returns:
            Словарь canonical_name -> user_id
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT canonical_name, user_id FROM profiles')
                rows = cursor.fetchall()
                return {row['canonical_name']: row['user_id'] for row in rows}
        except Exception as e:
            logger.error(f"Ошибка получения маппинга имен: {e}")
            return {}

    def clear_profiles(self, preserve_embeddings: bool = False) -> bool:
        """
        Очищает профили из базы данных

        Args:
            preserve_embeddings: Сохранять ли эмбеддинги

        Returns:
            True при успехе, False при ошибке
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM profiles')
                # Удаляем метаданные профилей
                cursor.execute("DELETE FROM metadata WHERE key NOT LIKE 'embedding_progress_%'")
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Ошибка очистки профилей: {e}")
            return False


class SettingsRepository:
    """Репозиторий для работы с настройками"""

    def save_setting(self, key: str, value: str) -> bool:
        """
        Сохраняет настройку

        Args:
            key: Ключ настройки
            value: Значение настройки

        Returns:
            True при успехе, False при ошибке
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO settings (key, value)
                    VALUES (?, ?)
                ''', (key, value))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Ошибка сохранения настройки {key}: {e}")
            return False

    def get_setting(self, key: str) -> Optional[str]:
        """
        Получает настройку по ключу

        Args:
            key: Ключ настройки

        Returns:
            Значение настройки или None
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT value FROM settings WHERE key = ?', (key,))
                row = cursor.fetchone()
                return row['value'] if row else None
        except Exception as e:
            logger.error(f"Ошибка получения настройки {key}: {e}")
            return None

    def get_all_settings(self) -> Dict[str, str]:
        """
        Получает все настройки

        Returns:
            Словарь настроек
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT key, value FROM settings')
                rows = cursor.fetchall()
                return {row['key']: row['value'] for row in rows}
        except Exception as e:
            logger.error(f"Ошибка получения всех настроек: {e}")
            return {}


class MetadataRepository:
    """Репозиторий для работы с метаданными"""

    def save_metadata(self, key: str, value: str) -> bool:
        """
        Сохраняет метаданные

        Args:
            key: Ключ метаданных
            value: Значение метаданных

        Returns:
            True при успехе, False при ошибке
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO metadata (key, value)
                    VALUES (?, ?)
                ''', (key, value))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Ошибка сохранения метаданных {key}: {e}")
            return False

    def get_metadata(self, key: str) -> Optional[str]:
        """
        Получает метаданные по ключу

        Args:
            key: Ключ метаданных

        Returns:
            Значение метаданных или None
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT value FROM metadata WHERE key = ?', (key,))
                row = cursor.fetchone()
                return row['value'] if row else None
        except Exception as e:
            logger.error(f"Ошибка получения метаданных {key}: {e}")
            return None


class MessageEmbeddingRepository:
    """Репозиторий для работы с эмбеддингами сообщений"""

    def save_embedding(
        self,
        canonical_name: str,
        message_text: str,
        embedding: Optional[List[float]] = None,
        source: str = "manual"
    ) -> bool:
        """
        Сохраняет эмбеддинг сообщения

        Args:
            canonical_name: Каноническое имя пользователя
            message_text: Текст сообщения
            embedding: Вектор эмбеддинга
            source: Источник эмбеддинга

        Returns:
            True при успехе, False при ошибке
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                ensure_message_embeddings_table(cursor)

                # Сериализуем эмбеддинг в BLOB
                embedding_blob = None
                if embedding is not None:
                    import pickle
                    embedding_blob = pickle.dumps(embedding)

                cursor.execute('''
                    INSERT INTO message_embeddings
                    (canonical_name, message_text, embedding, source)
                    VALUES (?, ?, ?, ?)
                ''', (canonical_name, message_text, embedding_blob, source))

                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Ошибка сохранения эмбеддинга для {canonical_name}: {e}")
            return False

    def get_embeddings_count(self, canonical_name: str) -> int:
        """
        Получает количество эмбеддингов для пользователя

        Args:
            canonical_name: Каноническое имя пользователя

        Returns:
            Количество эмбеддингов
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                ensure_message_embeddings_table(cursor)
                cursor.execute(
                    'SELECT COUNT(*) FROM message_embeddings WHERE canonical_name = ?',
                    (canonical_name,)
                )
                count = cursor.fetchone()[0]
                return count
        except Exception as e:
            logger.error(f"Ошибка получения счетчика эмбеддингов для {canonical_name}: {e}")
            return 0

    def is_message_processed(self, canonical_name: str, message_text: str) -> bool:
        """
        Проверяет, было ли обработано сообщение

        Args:
            canonical_name: Каноническое имя пользователя
            message_text: Текст сообщения

        Returns:
            True если сообщение уже обработано
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                ensure_message_embeddings_table(cursor)
                cursor.execute(
                    'SELECT COUNT(*) FROM message_embeddings WHERE canonical_name = ? AND message_text = ?',
                    (canonical_name, message_text)
                )
                count = cursor.fetchone()[0]
                return count > 0
        except Exception as e:
            logger.error(f"Ошибка проверки обработки сообщения для {canonical_name}: {e}")
            return False