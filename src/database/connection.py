"""
Подключение к базе данных SQLite
"""
import sqlite3
from contextlib import contextmanager
from typing import Generator
from pathlib import Path

from config.settings import settings
from utils import get_logger

logger = get_logger(__name__)


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Контекстный менеджер для подключения к базе данных

    Yields:
        Подключение к SQLite
    """
    conn = None
    try:
        # Создаем директорию для базы данных если её нет
        db_path = Path(settings.db_profiles)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(settings.db_profiles)
        conn.row_factory = sqlite3.Row  # Для доступа по имени столбца
        conn.execute("PRAGMA foreign_keys = ON")  # Включаем внешние ключи

        yield conn

    except Exception as e:
        logger.error(f"Ошибка подключения к базе данных: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def initialize_database() -> None:
    """
    Инициализация базы данных - создание всех таблиц
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Таблица профилей пользователей
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS profiles (
                    user_id INTEGER PRIMARY KEY,
                    canonical_name TEXT NOT NULL,
                    profile_text TEXT NOT NULL,
                    training_date TEXT NOT NULL,
                    version TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Таблица алиасов пользователей
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS aliases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    alias TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, alias),
                    FOREIGN KEY (user_id) REFERENCES profiles (user_id) ON DELETE CASCADE
                )
            ''')

            # Таблица эмбеддингов сообщений
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS message_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    canonical_name TEXT NOT NULL,
                    message_text TEXT NOT NULL,
                    embedding BLOB,
                    source TEXT DEFAULT 'manual',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Таблица метаданных
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Таблица настроек
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()
            logger.info("✓ База данных инициализирована успешно")

    except Exception as e:
        logger.error(f"Ошибка инициализации базы данных: {e}")
        raise


def ensure_message_embeddings_table(cursor: sqlite3.Cursor) -> None:
    """
    Гарантирует существование таблицы message_embeddings

    Args:
        cursor: Курсор базы данных
    """
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS message_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_name TEXT NOT NULL,
            message_text TEXT NOT NULL,
            embedding BLOB,
            source TEXT DEFAULT 'manual',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')


def cleanup_old_embeddings(days: int = 30) -> int:
    """
    Очищает старые эмбеддинги (старше N дней)

    Args:
        days: Количество дней для хранения

    Returns:
        Количество удаленных записей
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Удаляем старые эмбеддинги
            cursor.execute('''
                DELETE FROM message_embeddings
                WHERE created_at < datetime('now', '-{} days')
            '''.format(days))

            deleted_count = cursor.rowcount
            conn.commit()

            if deleted_count > 0:
                logger.info(f"✓ Очищено {deleted_count} старых эмбеддингов")

            return deleted_count

    except Exception as e:
        logger.error(f"Ошибка очистки старых эмбеддингов: {e}")
        return 0


def get_database_stats() -> dict:
    """
    Получает статистику базы данных

    Returns:
        Статистика по таблицам
    """
    stats = {}
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Получаем список таблиц
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()

            for table_row in tables:
                table_name = table_row['name']

                # Получаем количество записей
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]

                stats[table_name] = count

    except Exception as e:
        logger.error(f"Ошибка получения статистики базы данных: {e}")

    return stats