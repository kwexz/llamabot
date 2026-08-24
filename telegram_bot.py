import json
import re
import sqlite3
import requests
import os
import sys
import asyncio
import numpy as np
import pickle
import builtins
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from dotenv import load_dotenv
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Update
import chromadb
from chromadb.config import Settings

# Настройки Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# Модель будет загружена из базы данных или .env файла
OLLAMA_MODEL = None  # Будет инициализировано позже
EMBEDDING_MODEL = None  # Будет инициализировано позже
DB_PROFILES = os.getenv("DB_PROFILES", "user_profiles.db")
DB_EMBEDDINGS = os.getenv("DB_EMBEDDINGS", "./chroma_db")
PROFILES_VERSION = "1.0"  # Версия формата профилей
EMBEDDINGS_VERSION = "1.0"  # Версия формата эмбеддингов
OLLAMA_MAX_RETRIES = 1
OLLAMA_RETRY_BACKOFF = [0, 2, 5]  # seconds
LOG_DIR = os.getenv("LOG_DIR", "logs")

_ORIGINAL_PRINT = builtins.print


def _timestamped_print(*args, **kwargs):
    """Переопределенный print с добавлением timestamp и логированием в файл"""
    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")
    file = kwargs.pop("file", sys.stdout)
    flush = kwargs.pop("flush", False)

    message = sep.join(str(arg) for arg in args)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"{timestamp} {message}"

    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = os.path.join(LOG_DIR, f"{datetime.now():%Y-%m-%d}.log")
        with open(log_file, "a", encoding="utf-8") as log_f:
            log_f.write(formatted + end)
    except Exception:
        # Избегаем рекурсивного логирования ошибок записи
        pass

    _ORIGINAL_PRINT(formatted, file=file, end=end, flush=flush)


builtins.print = _timestamped_print


class TextAnalyzer:
    """Простой анализатор текста для вычисления частот слов."""

    WORD_REGEX = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)

    def __init__(self, messages: List[str]):
        self.messages = messages

    def most_common_words(self, top_n: int = 30) -> List[Tuple[str, int]]:
        counter: Counter = Counter()
        for msg in self.messages:
            for match in self.WORD_REGEX.findall(msg.lower()):
                if len(match) < 2:
                    continue
                counter[match] += 1
        return counter.most_common(top_n)

class TelegramBot:
    # Канонические имена и алиасы загружаются из базы данных
    CANONICAL_NAMES = []
    ALIASES = {}  # alias -> canonical_name
    
    def __init__(self, json_file: str = "result.json"):
        # Инициализируем базу данных для профилей
        self._init_database()

        # Загружаем канонические имена и маппинг из базы данных
        self._load_names_from_db()

        # Инициализируем ChromaDB
        self._init_chromadb()

        self.json_file = json_file
        self.messages_by_user: Dict[str, List[str]] = defaultdict(list)  # canonical_name -> messages
        self.user_profiles: Dict[str, str] = {}  # canonical_name -> profile
        self.user_names: Dict[str, str] = {}  # from_id -> canonical_name
        self.name_to_user_id: Dict[str, int] = {}  # canonical_name -> user_id
        self.name_variants: Dict[str, List[str]] = defaultdict(list)  # canonical_name -> [variants]
        self.telegram_app: Optional[Application] = None
        self.bot_username: Optional[str] = None  # Username бота для проверки упоминаний
        # Очередь запросов к Ollama
        self.ollama_queue: Optional[asyncio.Queue] = None
        self.ollama_processing = False  # Флаг обработки запроса
        self.max_queue_size = 5  # Максимальный размер очереди
        # Загружаем ID администратора из переменных окружения
        admin_user_id_str = os.getenv("ADMIN_USER_ID")
        self.admin_user_id = int(admin_user_id_str) if admin_user_id_str else None

    def _init_database(self):
        """Инициализирует базу данных SQLite для профилей"""
        try:
            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()

            # Создаем таблицу для профилей пользователей
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS profiles (
                    user_id integer PRIMARY KEY,
                    canonical_name TEXT NOT NULL,
                    profile_text TEXT NOT NULL,
                    training_date TEXT NOT NULL,
                    version TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Создаем таблицу для алиасов пользователей
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS aliases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    alias TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, alias)
                )
            ''')

            # Создаем таблицу для метаданных (дата обучения, версия и т.д.)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            ''')

            # Создаем таблицу для настроек (канонические имена и маппинг)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            ''')

            conn.commit()
            conn.close()
            print("✓ База данных SQLite инициализирована")
        except Exception as e:
            print(f"Ошибка при инициализации базы данных: {e}")

    def _normalize_name(self, name: str) -> Optional[str]:
        """Нормализует имя пользователя к каноническому варианту"""
        if not name:
            return None

        # Проверяем алиасы
        if name in self.ALIASES:
            return self.ALIASES[name]

        # Проверяем точное совпадение с каноническими именами
        if name in self.CANONICAL_NAMES:
            return name

        # Проверяем частичное совпадение (на случай небольших различий в написании)
        name_lower = name.lower().strip()
        for canonical in self.CANONICAL_NAMES:
            canonical_lower = canonical.lower().strip()
            # Проверяем, содержит ли имя каноническое имя или наоборот
            if name_lower == canonical_lower or name_lower in canonical_lower or canonical_lower in name_lower:
                return canonical

        return None

    def _save_user_aliases(self, user_id: int, aliases: List[str]):
        """Сохраняет алиасы пользователя в базу данных"""
        if not aliases:
            return

        try:
            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()

            # Вставляем алиасы (используем INSERT OR IGNORE для избежания дубликатов)
            for alias in aliases:
                if alias.strip():
                    cursor.execute('''
                        INSERT OR IGNORE INTO aliases (user_id, alias)
                        VALUES (?, ?)
                    ''', (user_id, alias.strip()))

            conn.commit()
            conn.close()
            print(f"✓ Сохранены алиасы для user_id {user_id}: {aliases}")
        except Exception as e:
            print(f"Ошибка при сохранении алиасов для user_id {user_id}: {e}")

    def _get_user_aliases(self, user_id: int) -> List[str]:
        """Получает все алиасы пользователя из базы данных"""
        try:
            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()
            cursor.execute('SELECT alias FROM aliases WHERE user_id = ?', (user_id,))
            rows = cursor.fetchall()
            conn.close()
            return [row[0] for row in rows]
        except Exception as e:
            print(f"Ошибка при получении алиасов для user_id {user_id}: {e}")
            return []

    def _find_user_by_alias(self, alias: str) -> Optional[str]:
        """Находит каноническое имя пользователя по алиасу"""
        try:
            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT canonical_name FROM profiles
                JOIN aliases ON profiles.user_id = aliases.user_id
                WHERE aliases.alias = ?
            ''', (alias.lower(),))
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else None
        except Exception as e:
            print(f"Ошибка при поиске пользователя по алиасу '{alias}': {e}")
            return None

    async def _update_all_user_aliases_from_telegram(self, bot):
        """Обновляет алиасы всех пользователей из Telegram API на основе существующих профилей"""
        try:
            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()
            cursor.execute('SELECT user_id FROM profiles')
            rows = cursor.fetchall()
            conn.close()

            for row in rows:
                user_id = row[0]
                try:
                    chat = await bot.get_chat(user_id)
                    # Извлекаем возможные алиасы из информации о пользователе
                    user_aliases = []
                    if getattr(chat, 'first_name', None):
                        user_aliases.append(chat.first_name)
                    if getattr(chat, 'last_name', None):
                        user_aliases.append(chat.last_name)
                    if getattr(chat, 'username', None):
                        user_aliases.append(chat.username)

                    # Убираем дубликаты
                    user_aliases = list(set(user_aliases))

                    if user_aliases:
                        # Получаем существующие алиасы
                        existing_aliases = self._get_user_aliases(user_id)
                        # Находим новые алиасы
                        new_aliases = [alias for alias in user_aliases if alias not in existing_aliases]
                        if new_aliases:
                            self._save_user_aliases(user_id, new_aliases)
                            print(f"✓ Обновлены алиасы для user_id {user_id}: {new_aliases}")
                        else:
                            print(f"[DEBUG] Нет новых алиасов для user_id {user_id}")

                except Exception as e:
                    print(f"Не удалось получить информацию о пользователе user_id {user_id}: {e}")
        except Exception as e:
            print(f"Ошибка при обновлении алиасов из Telegram API: {e}")
    
    def _is_allowed_user(self, name: str) -> bool:
        """Проверяет, является ли пользователь разрешенным"""
        return self._normalize_name(name) is not None
    
    def _init_chromadb(self):
        """Инициализирует ChromaDB для хранения эмбеддингов"""
        try:
            # Создаем клиент ChromaDB
            self.chroma_client = chromadb.PersistentClient(path=DB_EMBEDDINGS)

            # Создаем коллекцию для эмбеддингов сообщений
            try:
                self.message_embeddings_collection = self.chroma_client.get_collection("message_embeddings")
            except:
                self.message_embeddings_collection = self.chroma_client.create_collection(
                    "message_embeddings",
                    metadata={"hnsw:space": "cosine"}  # Используем косинусное расстояние для поиска
                )


            print("✓ ChromaDB инициализирована успешно")
            if self.message_embeddings_collection:
                total_embeddings = len(self.message_embeddings_collection.get()['ids'])
                print(f"[DEBUG] ChromaDB collection 'message_embeddings' has {total_embeddings} total embeddings")
        except Exception as e:
            print(f"Ошибка при инициализации ChromaDB: {e}")
            # Если ChromaDB не работает, продолжаем с SQLite
            self.chroma_client = None
            self.message_embeddings_collection = None

        
    def load_messages(self):
        """Загружает и парсит сообщения из JSON файла"""
        print("Загрузка сообщений из result.json...")
        print(f"Анализируются только сообщения участников: {', '.join(self.CANONICAL_NAMES)}")
        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        messages = data.get("messages", [])
        print(f"Найдено {len(messages)} сообщений (будут отфильтрованы только разрешенные участники)")
        
        for msg in messages:
            # Пропускаем служебные сообщения
            if msg.get("type") != "message":
                continue
            
            from_name = msg.get("from")
            from_id = msg.get("from_id")
            
            # Нормализуем имя к каноническому варианту
            canonical_name = self._normalize_name(from_name)
            if not canonical_name:
                # Пропускаем сообщения от неразрешенных участников
                continue
            
            # Сохраняем вариант имени для отображения
            if canonical_name not in self.name_variants or from_name not in self.name_variants[canonical_name]:
                self.name_variants[canonical_name].append(from_name)
            
            # Обрабатываем text (может быть строкой или массивом)
            text_raw = msg.get("text", "")
            if isinstance(text_raw, list):
                # Если text - массив, извлекаем только plain текст
                text_parts = []
                for item in text_raw:
                    if isinstance(item, dict):
                        if item.get("type") == "plain":
                            text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        # Если элемент - строка, добавляем её
                        text_parts.append(item)
                text = " ".join(text_parts).strip()
            else:
                text = str(text_raw).strip() if text_raw else ""
            
            # Извлекаем только plain текст из text_entities
            text_entities = msg.get("text_entities", [])
            plain_text_parts = []
            
            for entity in text_entities:
                if entity.get("type") == "plain":
                    plain_text_parts.append(entity.get("text", ""))
            
            # Если есть plain текст из entities, используем его, иначе используем text
            if plain_text_parts:
                final_text = " ".join(plain_text_parts).strip()
            elif text:
                final_text = text
            else:
                # Нет текстового содержимого, пропускаем
                continue
            
            # Пропускаем если текст пустой или слишком короткий (только ссылки/медиа)
            if not final_text or len(final_text) < 2:
                continue
            
            # Сохраняем сообщение под каноническим именем
            self.messages_by_user[canonical_name].append(final_text)
            # Сохраняем маппинг from_id -> canonical_name (для обратной совместимости)
            if from_id:
                self.user_names[from_id] = canonical_name
                # Создаем обратный маппинг canonical_name -> user_id
                user_id_clean = self._get_user_id_from_canonical_name(canonical_name, from_id)
                self.name_to_user_id[canonical_name] = user_id_clean
        
        print(f"Обработано сообщений для {len(self.messages_by_user)} участников:")
        for canonical_name, messages in self.messages_by_user.items():
            count = len(messages)
            variants = self.name_variants.get(canonical_name, [])
            if len(variants) > 1:
                print(f"  {canonical_name} (варианты: {', '.join(variants)}): {count} сообщений")
            else:
                print(f"  {canonical_name}: {count} сообщений")
    
    def _call_ollama_sync(self, prompt: str) -> str:
        """Синхронный вызов Ollama API для генерации ответа"""
        # Убеждаемся, что модель загружена
        if OLLAMA_MODEL is None:
            self._load_ollama_model()

        for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
            try:
                response = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=300
                )
                response.raise_for_status()
                result = response.json()    
                return result.get("response", "").strip()
            except Exception as e:
                error_msg = f"Ошибка при обращении к Ollama (попытка {attempt}/{OLLAMA_MAX_RETRIES}): {e}"
                print(error_msg)
                if attempt == OLLAMA_MAX_RETRIES:
                    return error_msg
                backoff = OLLAMA_RETRY_BACKOFF[min(attempt, len(OLLAMA_RETRY_BACKOFF) - 1)]
                if backoff > 0:
                    time.sleep(backoff)
    
    def _get_embedding_sync(self, text: str) -> Optional[np.ndarray]:
        """Синхронный вызов Ollama API для получения эмбеддинга"""
        # Убеждаемся, что модель загружена
        if OLLAMA_MODEL is None:
            self._load_ollama_model()

        for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
            try:
                response = requests.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": text
                    },
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                embedding = result.get("embedding")
                if embedding:
                    return np.array(embedding, dtype=np.float32)
                return None
            except Exception as e:
                print(f"Ошибка при получении эмбеддинга (попытка {attempt}/{OLLAMA_MAX_RETRIES}): {e}")
                if attempt == OLLAMA_MAX_RETRIES:
                    return None
                backoff = OLLAMA_RETRY_BACKOFF[min(attempt, len(OLLAMA_RETRY_BACKOFF) - 1)]
                if backoff > 0:
                    time.sleep(backoff)

    def _get_ollama_models_sync(self) -> List[str]:
        """Синхронный вызов Ollama API для получения списка локальных моделей"""
        for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
            try:
                response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=30)
                response.raise_for_status()
                result = response.json()
                models = result.get("models", [])
                model_names = [model.get("name", "") for model in models if model.get("name")]
                return model_names
            except Exception as e:
                print(f"Ошибка при получении списка моделей Ollama (попытка {attempt}/{OLLAMA_MAX_RETRIES}): {e}")
                if attempt == OLLAMA_MAX_RETRIES:
                    return []
                backoff = OLLAMA_RETRY_BACKOFF[min(attempt, len(OLLAMA_RETRY_BACKOFF) - 1)]
                if backoff > 0:
                    time.sleep(backoff)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Вычисляет косинусное сходство между двумя векторами"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    
    def _check_embeddings_exist(self, canonical_name: str) -> bool:
        """Проверяет, существуют ли эмбеддинги для участника"""
        try:
            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()
            self._ensure_embeddings_table(cursor)
            conn.commit()
            cursor.execute(
                'SELECT COUNT(*) FROM message_embeddings WHERE canonical_name = ?',
                (canonical_name,)
            )
            count = cursor.fetchone()[0]
            conn.close()
            return count > 0
        except Exception as e:
            print(f"Ошибка при проверке эмбеддингов: {e}")
            return False

    def _get_processed_embeddings_count(self, canonical_name: str) -> int:
        """Получает количество обработанных эмбеддингов для участника"""
        try:
            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()
            cursor.execute(
                'SELECT COUNT(*) FROM message_embeddings WHERE canonical_name = ?',
                (canonical_name,)
            )
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            print(f"Ошибка при получении счетчика эмбеддингов: {e}")
            return 0

    def _save_embedding_progress(self, canonical_name: str, processed_count: int):
        """Сохраняет прогресс создания эмбеддингов"""
        try:
            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO metadata (key, value)
                VALUES (?, ?)
            ''', (f"embedding_progress_{canonical_name}", str(processed_count)))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Ошибка при сохранении прогресса эмбеддингов: {e}")

    def _get_embedding_progress(self, canonical_name: str) -> int:
        """Получает сохраненный прогресс создания эмбеддингов"""
        try:
            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()
            cursor.execute(
                'SELECT value FROM metadata WHERE key = ?',
                (f"embedding_progress_{canonical_name}",)
            )
            row = cursor.fetchone()
            conn.close()
            return int(row[0]) if row else 0
        except Exception as e:
            print(f"Ошибка при получении прогресса эмбеддингов: {e}")
            return 0

    def _clear_embedding_progress(self, canonical_name: str):
        """Очищает сохраненный прогресс создания эмбеддингов"""
        try:
            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()
            cursor.execute(
                'DELETE FROM metadata WHERE key = ?',
                (f"embedding_progress_{canonical_name}",)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Ошибка при очистке прогресса эмбеддингов: {e}")
    

    def _create_all_embeddings_chromadb(self, force_recreate: bool = False):
        """Создает эмбеддинги для всех участников в ChromaDB с поддержкой прерываний и возобновления"""
        print("\nСоздание базы знаний в ChromaDB (эмбеддинги сообщений)...")

        for canonical_name in self.messages_by_user.keys():
            # Используем правильный user_id из маппинга
            user_id = self.name_to_user_id.get(canonical_name)
            if not user_id:
                print(f"⚠ Пропускаем {canonical_name}: user_id не найден")
                continue
            self._create_embeddings_for_user_chromadb(user_id, canonical_name, force_recreate)
        print("✓ База знаний в ChromaDB создана\n")

    def _create_embeddings_for_user_chromadb(self, user_id: int, canonical_name: str, force_recreate: bool = False):
        """Создает и сохраняет эмбеддинги для всех сообщений участника в ChromaDB"""
        messages = self.messages_by_user.get(canonical_name, [])
        if not messages:
            return

        print(f"  Создание эмбеддингов для {canonical_name} ({user_id}) ({len(messages)} сообщений)...")

        try:
            # Подсчет уже существующих эмбеддингов для этого пользователя
            existing_count = 0
            if self.message_embeddings_collection:
                try:
                    results = self.message_embeddings_collection.get(where={"user_id": user_id})
                    existing_count = len(results['ids'])
                except Exception as e:
                    print(f"[DEBUG] Ошибка при подсчете существующих эмбеддингов: {e}")
                    existing_count = 0

            print(f"[DEBUG] Для {canonical_name} уже существует {existing_count} эмбеддингов из {len(messages)} сообщений")
            if existing_count >= len(messages) and not force_recreate:
                print(f"  Эмбеддинги для {canonical_name} уже существуют ({existing_count}/{len(messages)}), пропускаем")
                return

            # Создаем эмбеддинги для сообщений
            created = 0
            for i, message in enumerate(messages):
                if (i + 1) % 50 == 0:
                    print(f"    Обработано {i + 1}/{len(messages)} сообщений...")

                # Проверяем, не обработано ли сообщение уже
                if not force_recreate and self._is_message_already_processed(user_id, message):
                    continue

                embedding = self._get_embedding_sync(message)
                if embedding is not None:
                    # Генерируем уникальный ID для документа
                    import hashlib
                    doc_id = hashlib.md5(f"{user_id}_{message}_{time.time()}".encode()).hexdigest()

                    # Сохраняем в ChromaDB
                    self.message_embeddings_collection.add(
                        documents=[message],
                        embeddings=[embedding.tolist()],
                        ids=[doc_id],
                        metadatas=[{
                            "user_id": user_id,
                            "canonical_name": canonical_name,
                            "message_text": message,
                            "timestamp": datetime.now().isoformat()
                        }]
                    )
                    created += 1

            print(f"  ✓ Создано {created} эмбеддингов для {canonical_name} (user_id: {user_id})")
        except Exception as e:
            print(f"  ✗ Ошибка при создании эмбеддингов для {canonical_name} ({user_id}): {e}")
    
    def _retrieve_relevant_messages(self, user_id: int, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Находит наиболее релевантные сообщения участника для запроса (RAG) из ChromaDB"""
        try:
            # Получаем эмбеддинг запроса
            query_embedding = self._get_embedding_sync(query)
            if query_embedding is None:
                return []

            # Поиск в ChromaDB
            if self.message_embeddings_collection:
                results = self.message_embeddings_collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k,
                    where={"user_id": user_id}  # Фильтруем по конкретному пользователю
                )

                # Обрабатываем результаты
                relevant_messages = []
                if results['documents'] and results['distances']:
                    for doc, distance in zip(results['documents'][0], results['distances'][0]):
                        # Преобразуем евклидово расстояние в сходство (чем меньше расстояние, тем больше сходство)
                        similarity = 1 / (1 + distance)  # Преобразуем расстояние в сходство
                        relevant_messages.append((doc, similarity))

                return relevant_messages
            else:
                print("ChromaDB недоступна для поиска релевантных сообщений")
                return []
        except Exception as e:
            print(f"Ошибка при поиске релевантных сообщений: {e}")
            return []

    def _print_embeddings_stats(self):
        """Выводит статистику эмбеддингов для каждого участника"""
        print("\n📊 Статистика эмбеддингов:")

        if not self.message_embeddings_collection:
            print("ChromaDB недоступна")
            return

        for canonical_name, user_id in self.name_to_user_id.items():
            try:
                results = self.message_embeddings_collection.get(where={"user_id": user_id})
                embeddings_count = len(results['ids'])
                messages_count = len(self.messages_by_user.get(canonical_name, []))
                print(f"  {canonical_name}: {embeddings_count}/{messages_count} эмбеддингов (user_id: {user_id})")
            except Exception as e:
                print(f"  Ошибка при получении статистики для {canonical_name}: {e}")

    
    async def call_ollama(self, prompt: str) -> str:
        """Вызывает Ollama API через очередь для генерации ответа"""
        # Инициализируем очередь при первом использовании
        if self.ollama_queue is None:
            self.ollama_queue = asyncio.Queue()
            # Запускаем обработчик очереди
            try:
                asyncio.create_task(self._process_ollama_queue())
            except RuntimeError:
                # Если event loop не запущен, запустим обработчик позже
                pass
        
        # Проверяем размер очереди (включая текущий обрабатываемый запрос)
        queue_size = self.ollama_queue.qsize()
        if self.ollama_processing:
            queue_size += 1  # Учитываем текущий обрабатываемый запрос
        
        if queue_size >= self.max_queue_size:
            return "⏳ Слишком много запросов в очереди. Пожалуйста, повторите позже, когда модель освободится."
        
        # Создаем Future для получения результата
        future = asyncio.Future()
        
        # Добавляем запрос в очередь
        await self.ollama_queue.put((prompt, future))
        queue_size_after = self.ollama_queue.qsize()
        if self.ollama_processing:
            queue_size_after += 1
        print(f"[Ollama] Запрос добавлен в очередь (позиция: {queue_size_after}/{self.max_queue_size})")
        
        # Ждем результат
        try:
            result = await future
            return result
        except Exception as e:
            return f"Ошибка при обработке запроса: {e}"
    
    async def _process_ollama_queue(self):
        """Обрабатывает очередь запросов к Ollama последовательно"""
        print("[Ollama] Обработчик очереди запущен")
        while True:
            try:
                # Ждем запрос из очереди
                prompt, future = await self.ollama_queue.get()
                
                print(f"[Ollama] Начало обработки запроса (осталось в очереди: {self.ollama_queue.qsize()})")
                self.ollama_processing = True
                
                try:
                    # Выполняем запрос в отдельном потоке, чтобы не блокировать event loop
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, self._call_ollama_sync, prompt)
                    
                    # Устанавливаем результат
                    if not future.done():
                        future.set_result(result)
                    print(f"[Ollama] Запрос обработан успешно")
                except Exception as e:
                    error_msg = f"Ошибка при обращении к Ollama: {e}"
                    if not future.done():
                        future.set_exception(Exception(error_msg))
                    print(f"[Ollama] Ошибка при обработке запроса: {e}")
                finally:
                    self.ollama_processing = False
                    self.ollama_queue.task_done()
                    
            except Exception as e:
                print(f"[Ollama] Критическая ошибка в обработчике очереди: {e}")
                await asyncio.sleep(1)  # Небольшая задержка перед следующей попыткой
    
    def create_profile(self, canonical_name: str) -> str:
        """Создает портрет участника на основе его сообщений"""
        messages = self.messages_by_user[canonical_name]

        if not messages:
            return ""

        # 1) Фильтр технических/пустых сообщений
        technical_phrases = {
            "ок",
            "окей",
            "спасибо",
            "спс",
            "thx",
            "ок.",
            "ок!",
            "спасибо!",
            "спасибо.",
        }

        def is_technical(msg: str) -> bool:
            text = msg.strip().lower()
            if not text:
                return True
            if text in technical_phrases:
                return True
            # очень короткие сообщения без содержательной нагрузки
            if len(text) <= 2:
                return True
            return False

        filtered = [m for m in messages if not is_technical(m)]

        if not filtered:
            filtered = messages

        # 2) Убираем дубликаты
        seen = set()
        unique_messages: List[str] = []
        for m in filtered:
            if m not in seen:
                seen.add(m)
                unique_messages.append(m)

        # 3) Сортируем по длине (самые развернутые в начало)
        unique_messages.sort(key=lambda x: len(x), reverse=True)

        # 4) Берем топ-50 сообщений
        selected_messages = unique_messages[:50]

        # 5) Выделяем самые частотные слова через TextAnalyzer
        analyzer = TextAnalyzer(selected_messages)
        common_words = analyzer.most_common_words(top_n=30)
        common_words_str = "\n".join(
            [f"- {word} ({count})" for word, count in common_words]
        ) or "- (недостаточно данных)"

        # 6) Формируем примеры сообщений для промпта
        examples = "\n".join([f"- {msg}" for msg in selected_messages])

        prompt = f"""[РОЛЬ]
Ты — лингвист-аналитик. Проанализируй репрезентативную выборку из истории сообщений пользователя {canonical_name} и создай стилевой портрет.

[ИСХОДНЫЕ ДАННЫЕ]
- Всего сообщений в выборке: {len(selected_messages)}
- Сообщения отсортированы по длине (самые развернутые в начале).
- Удалены технические короткие реплики ("ок", "спасибо" и т.п.).
- Убраны точные дубликаты фраз.

[НАИБОЛЕЕ ЧАСТО УПОМИНАЕМЫЕ СЛОВА/ФРАЗЫ]
{common_words_str}

[ПРЕДСТАВИТЕЛЬНЫЕ СООБЩЕНИЯ]
{examples}

[ФОКУС АНАЛИЗА]
Выдели устойчивые паттерны, которые повторяются в разных типах сообщений и временных периодах.

[УПРОЩЕННЫЙ ФОРМАТ ВЫВОДА]
- **Ядро стиля:** 3-5 ключевых характеристик
- **Лексические паттерны:** топ-10 часто используемых слов/фраз (можно ссылаться на список выше)
- **Синтаксические привычки:** структура предложений и абзацев
- **Коммуникативные тактики:** как ведет диалог
- **Готовые шаблоны:** 5-7 типичных для пользователя конструкций
"""
        # Для создания профилей используем синхронный вызов (это делается один раз при инициализации)
        profile = self._call_ollama_sync(prompt)
        return profile
    
    def load_profiles(self) -> bool:
        """Загружает сохраненные портреты из базы данных"""
        if not os.path.exists(DB_PROFILES):
            return False

        try:
            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()

            # Загружаем профили
            cursor.execute('SELECT user_id, canonical_name, profile_text, training_date, version FROM profiles')
            rows = cursor.fetchall()

            if rows:
                for user_id, canonical_name, profile_text, training_date, version in rows:
                    self.user_profiles[canonical_name] = profile_text
                    # Восстанавливаем маппинг user_id -> canonical_name (преобразуем в int для ChromaDB)
                    self.name_to_user_id[canonical_name] = int(user_id)
                    # Инициализируем пустой список сообщений для этого участника (из базы не загружаем сообщения)
                    if canonical_name not in self.messages_by_user:
                        self.messages_by_user[canonical_name] = []

                    # Загружаем алиасы для каждого участника
                    aliases = self._get_user_aliases(user_id)
                    if aliases:
                        # Сохраняем алиасы в памяти для быстрого поиска
                        for alias in aliases:
                            if alias not in self.name_variants[canonical_name]:
                                self.name_variants[canonical_name].append(alias)

                # Получаем общую дату обучения из метаданных или первой записи
                cursor.execute('SELECT value FROM metadata WHERE key = ?', ('training_date',))
                meta_date = cursor.fetchone()
                if meta_date:
                    training_date = meta_date[0]
                elif rows:
                    training_date = rows[0][2]
                else:
                    training_date = "неизвестно"

                cursor.execute('SELECT value FROM metadata WHERE key = ?', ('version',))
                meta_version = cursor.fetchone()
                version = meta_version[0] if meta_version else PROFILES_VERSION

                print(f"Загружено {len(self.user_profiles)} сохраненных портретов из базы данных")
                print(f"Дата обучения: {training_date}, версия: {version}")
                conn.close()
                return True
            else:
                conn.close()
                return False
        except Exception as e:
            print(f"Ошибка при загрузке портретов из базы данных: {e}")
            return False
    
    def save_profiles(self):
        """Сохраняет портреты в базу данных SQLite"""
        try:
            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()
            
            training_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Сохраняем или обновляем профили
            for canonical_name, profile_text in self.user_profiles.items():
                user_id = self.name_to_user_id.get(canonical_name)
                if user_id:
                    cursor.execute('''
                        INSERT OR REPLACE INTO profiles
                        (user_id, canonical_name, profile_text, training_date, version, updated_at)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (user_id, canonical_name, profile_text, training_date, PROFILES_VERSION))
            
            # Сохраняем метаданные
            cursor.execute('''
                INSERT OR REPLACE INTO metadata (key, value)
                VALUES ('training_date', ?)
            ''', (training_date,))
            
            cursor.execute('''
                INSERT OR REPLACE INTO metadata (key, value)
                VALUES ('version', ?)
            ''', (PROFILES_VERSION,))
            
            conn.commit()
            conn.close()
            print(f"Портреты сохранены в базу данных {DB_PROFILES}")
        except Exception as e:
            print(f"Ошибка при сохранении портретов в базу данных: {e}")
    
    def _clear_profiles(self, preserve_embeddings: bool = False):
        """Очищает профили из базы данных"""
        try:
            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM profiles')
            # Удаляем метаданные профилей
            cursor.execute("DELETE FROM metadata WHERE key NOT LIKE 'embedding_progress_%'")
            # Эмбеддинги в ChromaDB не затрагиваем
            conn.commit()
            conn.close()
            print("Профили очищены из базы данных")
        except Exception as e:
            print(f"Ошибка при очистке базы данных: {e}")
    
    def create_all_profiles(self, force_recreate: bool = False, rebuild_embeddings: bool = False):
        """Создает портреты для всех участников"""
        # Пытаемся загрузить сохраненные портреты
        profiles_loaded = False
        if not force_recreate:
            profiles_loaded = self.load_profiles()
            if profiles_loaded:
                # Проверяем, что все текущие участники имеют портреты
                missing_profiles = [name for name in self.messages_by_user.keys()
                                   if name not in self.user_profiles]
                if missing_profiles:
                    print(f"\nВнимание: найдены новые участники без портретов: {', '.join(missing_profiles)}")
                    print("Создаю портреты только для новых участников...")
                    # Создаем портреты только для отсутствующих
                    for canonical_name in missing_profiles:
                        print(f"Анализ стиля {canonical_name}...")
                        profile = self.create_profile(canonical_name)
                        self.user_profiles[canonical_name] = profile
                        print(f"✓ {canonical_name}: {profile[:100]}...")
                    # Сохраняем обновленные портреты
                    self.save_profiles()
                    if rebuild_embeddings:
                        # Запускаем создание эмбеддингов в фоне
                        try:
                            import threading
                            embedding_thread = threading.Thread(target=self._create_all_embeddings_background, args=(False,))
                            embedding_thread.daemon = True
                            embedding_thread.start()
                            print("🚀 Создание эмбеддингов запущено в фоне после профилей...")
                        except Exception as e:
                            print(f"Ошибка при запуске фонового создания эмбеддингов: {e}")
                    return

        # Если портреты не загружены или требуется пересоздание
        if force_recreate or not profiles_loaded:
            if force_recreate:
                print("\nПринудительное пересоздание портретов...")
                # Очищаем старые профили, при необходимости сохраняя эмбеддинги
                preserve_embeddings = not rebuild_embeddings
                self._clear_profiles(preserve_embeddings=preserve_embeddings)
                self.user_profiles = {}
            else:
                print("\nСоздание портретов участников (первый запуск)...")

            for canonical_name in self.messages_by_user.keys():
                print(f"Анализ стиля {canonical_name}...")
                profile = self.create_profile(canonical_name)
                self.user_profiles[canonical_name] = profile
                print(f"✓ {canonical_name}: {profile[:100]}...")

            # Сохраняем портреты
            self.save_profiles()
        else:
            print("\n✓ Используются сохраненные портреты участников (обучение уже выполнено)")

        # Создаем/обновляем базу знаний (эмбеддинги) для RAG в фоне
        if rebuild_embeddings:
            try:
                import threading
                embedding_thread = threading.Thread(target=self._create_all_embeddings_background, args=(force_recreate,))
                embedding_thread.daemon = True
                embedding_thread.start()
                print("🚀 Создание эмбеддингов запущено в фоне...")
            except Exception as e:
                print(f"Ошибка при запуске фонового создания эмбеддингов: {e}")
        else:
            print("ℹ Пересоздание эмбеддингов пропущено (запрошено обновление только профилей)")

    def _create_all_embeddings_background(self, force_recreate: bool = False):
        """Создает эмбеддинги для всех участников в фоновом потоке"""
        try:
            # Создаем эмбеддинги только в ChromaDB
            self._create_all_embeddings_chromadb(force_recreate)
        except Exception as e:
            print(f"Ошибка при фоновом создании эмбеддингов: {e}")
    
    def find_user_by_name(self, name: str) -> Optional[str]:
        """Находит каноническое имя по точному совпадению, включая алиасы"""
        # Нормализуем имя
        canonical = self._normalize_name(name)
        if canonical and canonical in self.messages_by_user:
            return canonical

        # Ищем по алиасам в базе данных (точное совпадение)
        alias_match = self._find_user_by_alias(name)
        if alias_match and alias_match in self.messages_by_user:
            return alias_match

        # Если не найдено в messages_by_user, ищем в загруженных профилях из базы данных
        # Это позволяет использовать участников, которые есть в базе, но возможно отсутствуют в свежем JSON
        if canonical and canonical in self.user_profiles:
            return canonical

        return None
    
    async def generate_response(self, question: str, canonical_name: str, verbose: bool = False) -> str:
        """Генерирует ответ в стиле указанного участника с использованием RAG"""
        profile = self.user_profiles.get(canonical_name, "")

        # Получаем user_id для RAG поиска
        user_id = self.name_to_user_id.get(canonical_name)
        if not user_id:
            # Fallback на старый метод
            user_id = self._get_user_id_from_canonical_name(canonical_name, self.user_names.get(canonical_name, canonical_name))

        # Используем RAG для поиска релевантных сообщений
        print(f"[RAG] Поиск релевантных сообщений для вопроса: {question[:50]}...")
        relevant_messages = self._retrieve_relevant_messages(user_id, question, top_k=5)

        if relevant_messages:
            print(f"[RAG] Найдено {len(relevant_messages)} релевантных сообщений")
            logged_messages = []
            for idx, (msg, score) in enumerate(relevant_messages, 1):
                snippet = msg.strip()
                if len(snippet) > 400:
                    snippet = snippet[:400] + "..."
                print(f"[RAG][{canonical_name}] #{idx} (score {score:.3f}): {snippet}")
                logged_messages.append(f"- {msg}")
            context_messages = "\n".join(logged_messages)
            context_info = f"Релевантные сообщения {canonical_name} по теме:\n{context_messages}"
        else:
            print(f"[RAG] Релевантные сообщения не найдены, используем последние сообщения")
            # Если RAG не нашел релевантных сообщений, используем последние
            messages = self.messages_by_user.get(canonical_name, [])
            context_messages = "\n".join([f"- {msg}" for msg in messages[-10:]])
            context_info = f"Последние сообщения {canonical_name}:\n{context_messages}"

        prompt = f"""[РОЛЬ]
Ты принимаешь личность и стиль общения {canonical_name}. Вот твой портрет:
{profile}

[КОНТЕКСТ]
Ситуация, в которой происходит общение: {context_info}

[ЗАДАЧА]
Ответить на следующий вопрос, полностью сохраняя стиль {canonical_name}:
"{question}"

[ТРЕБОВАНИЯ К ОТВЕТУ]
1. **Содержание:** Ответ должен быть по существу вопроса.
2. **Стиль:** Должен быть естественным и неузнаваемо вжиться в роль. Используй характерные для {canonical_name}:
    *   **Словарный запас и жаргон:** Специфические слова, ругательства, профессиональные термины.
    *   **Речевые паттерны:** Длина предложений (короткие/длинные), риторические приемы, повторы.
    *   **Интонация и эмоции:** Самоуверенность, ирония, добродушие, грусть и т.д.
    *   **Уникальные выражения-фишки:** Известные фразы или "пунктики" персонажа.
3. **Структура:** Ответ должен быть целостным, не нужно помечать его как "Ответ:" или добавлять мета-комментарии."""

        response = await self.call_ollama(prompt)

        # Если включен verbose режим, добавляем релевантные сообщения к ответу
        if verbose and relevant_messages:
            verbose_info = "\n\n--- РЕЛЕВАНТНЫЕ СООБЩЕНИЯ ---\n"
            for idx, (msg, score) in enumerate(relevant_messages, 1):
                verbose_info += f"{idx}. [score: {score:.3f}] {msg}\n"
            response += verbose_info

        return response
    
    def parse_question(self, user_input: str) -> tuple:
        """Парсит вопрос извлекает имя участника из формата ^Имя^ и флаг verbose (различные вариации)"""
        # Ищем паттерн ^Имя^
        pattern = r'\^([^^]+)\^'
        matches = re.findall(pattern, user_input)

        # Проверяем флаг verbose с различными вариациями
        verbose = False
        verbose_variants = [
            '---verbose', # тройной дефис
            '-- verbose',  # двойной дефис с пробелом
            '—verbose',   # длинное тире
            ' —verbose',  # с пробелом перед длинным тире
            '--verbose',   # двойной дефис без пробела
            '-v'
        ]

        original_input = user_input
        for variant in verbose_variants:
            if variant.lower() in user_input.lower():
                verbose = True
                user_input = user_input.replace(variant, '', 1)  # заменяем только первое вхождение

        # Убираем лишние пробелы
        user_input = ' '.join(user_input.split())

        if matches:
            # Берем последнее совпадение
            name = matches[-1]
            # Удаляем ^Имя^ из вопроса
            question = re.sub(pattern, '', user_input).strip()
            return question, name, verbose
        else:
            return user_input, None, verbose
    
    def _load_allowed_chat_ids(self) -> List[int]:
        """Загружает список разрешенных ID чатов из базы данных"""
        try:
            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM settings WHERE key = ?', ('allowed_chat_ids',))
            row = cursor.fetchone()
            conn.close()

            if row:
                allowed_chats_str = row[0]
                # Парсим список ID чатов через запятую
                allowed_chat_ids = [int(chat_id.strip()) for chat_id in allowed_chats_str.split(",") if chat_id.strip()]
                return allowed_chat_ids
            else:
                # Если данных в БД нет, проверяем .env для обратной совместимости
                allowed_chats_str = os.getenv("ALLOWED_CHAT_IDS", "")
                if allowed_chats_str:
                    print("ℹ Используются ALLOWED_CHAT_IDS из .env файла (рекомендуется переместить в базу данных)")
                    allowed_chat_ids = [int(chat_id.strip()) for chat_id in allowed_chats_str.split(",") if chat_id.strip()]
                    # Сохраняем в базу для будущих запусков
                    self._save_allowed_chat_ids(allowed_chat_ids)
                    return allowed_chat_ids
                return []
        except Exception as e:
            print(f"Ошибка при загрузке разрешенных ID чатов: {e}")
            return []

    def _save_allowed_chat_ids(self, allowed_chat_ids: List[int]):
        """Сохраняет список разрешенных ID чатов в базу данных"""
        try:
            allowed_chats_str = ",".join(str(chat_id) for chat_id in allowed_chat_ids)
            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO settings (key, value)
                VALUES ('allowed_chat_ids', ?)
            ''', (allowed_chats_str,))
            conn.commit()
            conn.close()
            print(f"✓ Разрешенные ID чатов сохранены: {allowed_chats_str}")
        except Exception as e:
            print(f"Ошибка при сохранении разрешенных ID чатов: {e}")

    def _load_ollama_model(self) -> str:
        """Загружает модель Ollama из базы данных"""
        global OLLAMA_MODEL
        try:
            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM settings WHERE key = ?', ('ollama_model',))
            row = cursor.fetchone()
            conn.close()

            if row:
                OLLAMA_MODEL = row[0]
                return OLLAMA_MODEL
            else:
                # Если данных в БД нет, проверяем .env для обратной совместимости
                ollama_model = os.getenv("OLLAMA_MODEL", "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest")
                if ollama_model:
                    print("ℹ Используется OLLAMA_MODEL из .env файла (рекомендуется переместить в базу данных)")
                    # Сохраняем в базу для будущих запусков
                    self._save_ollama_model(ollama_model)
                    OLLAMA_MODEL = ollama_model
                    return ollama_model
                return "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest"
        except Exception as e:
            print(f"Ошибка при загрузке модели Ollama: {e}")
            return "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest"

    def _load_ollama_embedding_model(self) -> str:
        """Загружает модель эмбеддингов Ollama из базы данных"""
        global EMBEDDING_MODEL
        try:
            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM settings WHERE key = ?', ('ollama_embedding_model',))
            row = cursor.fetchone()
            conn.close()

            if row:
                EMBEDDING_MODEL = row[0]
                return EMBEDDING_MODEL
            else:
                # Если данных в БД нет, проверяем .env для обратной совместимости
                embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
                if embedding_model:
                    print("ℹ Используется OLLAMA_EMBEDDING_MODEL из .env файла (рекомендуется переместить в базу данных)")
                    # Сохраняем в базу для будущих запусков
                    self._save_ollama_embedding_model(embedding_model)
                    EMBEDDING_MODEL = embedding_model
                    return embedding_model
                return "nomic-embed-text"
        except Exception as e:
            print(f"Ошибка при загрузке модели эмбеддингов Ollama: {e}")
            return "nomic-embed-text"

    def _save_ollama_model(self, model_name: str):
        """Сохраняет модель Ollama в базу данных"""
        global OLLAMA_MODEL
        try:
            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO settings (key, value)
                VALUES ('ollama_model', ?)
            ''', (model_name,))
            conn.commit()
            conn.close()
            OLLAMA_MODEL = model_name
            print(f"✓ Модель Ollama сохранена: {model_name}")
        except Exception as e:
            print(f"Ошибка при сохранении модели Ollama: {e}")

    def _save_ollama_embedding_model(self, model_name: str):
        """Сохраняет модель эмбеддингов Ollama в базу данных"""
        global EMBEDDING_MODEL
        try:
            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO settings (key, value)
                VALUES ('ollama_embedding_model', ?)
            ''', (model_name,))
            conn.commit()
            conn.close()
            EMBEDDING_MODEL = model_name
            print(f"✓ Модель эмбеддингов Ollama сохранена: {model_name}")
        except Exception as e:
            print(f"Ошибка при сохранении модели эмбеддингов Ollama: {e}")

    def _is_allowed_chat(self, chat_id: int) -> bool:
        """
        Проверяет, разрешен ли чат/группа/пользователь для работы с ботом.

        Args:
            chat_id: ID чата, группы или пользователя (для личных сообщений)

        Returns:
            True если чат/группа/пользователь разрешен, False иначе
        """
        allowed_chat_ids = self._load_allowed_chat_ids()

        if not allowed_chat_ids:
            # Если список пустой, разрешаем все чаты/группы/пользователей
            return True

        is_allowed = chat_id in allowed_chat_ids

        # Отладочная информация
        if not is_allowed:
            print(f"[DEBUG] Чат ID {chat_id} не найден в списке разрешенных: {allowed_chat_ids}")
        # else:
            # print(f"[DEBUG] Чат ID {chat_id} разрешен")

        return is_allowed

    def _load_names_from_db(self):
        """Загружает канонические имена и маппинг из базы данных"""
        try:
            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()

            # Загружаем канонические имена
            cursor.execute('SELECT value FROM settings WHERE key = ?', ('canonical_names',))
            canonical_names_row = cursor.fetchone()
            if canonical_names_row:
                self.CANONICAL_NAMES = [name.strip() for name in canonical_names_row[0].split(",") if name.strip()]
            else:
                # Инициализируем пустой список, если данных нет в БД
                self.CANONICAL_NAMES = []

            # Строим маппинг имен из таблицы алиасов (alias -> canonical_name)
            cursor.execute('''
                SELECT aliases.alias, profiles.canonical_name
                FROM aliases
                JOIN profiles ON aliases.user_id = profiles.user_id
            ''')
            rows = cursor.fetchall()
            self.NAME_ALIASES = {alias: canonical for alias, canonical in rows}

            conn.close()
            print(f"✓ Загружено {len(self.CANONICAL_NAMES)} канонических имен и {len(self.ALIASES)} алиасов из базы данных")
        except Exception as e:
            print(f"Ошибка при загрузке настроек из базы данных: {e}")
            # Fallback на значения из .env файла
            self._load_names_from_env_fallback()

    def _load_names_from_env_fallback(self):
        """Загружает канонические имена и маппинг из .env файла (fallback)"""
        # Загружаем канонические имена
        canonical_names_str = os.getenv("CANONICAL_NAMES", "")
        if canonical_names_str:
            self.CANONICAL_NAMES = [name.strip() for name in canonical_names_str.split(",") if name.strip()]
        else:
            self.CANONICAL_NAMES = []

        # Алиасы загружаются из базы данных, fallback не используется
        self.ALIASES = {}

    def _save_canonical_names(self, names_str: str):
        """Сохраняет канонические имена в базу данных"""
        try:
            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO settings (key, value)
                VALUES ('canonical_names', ?)
            ''', (names_str,))
            conn.commit()
            conn.close()

            # Обновляем в памяти
            self.CANONICAL_NAMES = [name.strip() for name in names_str.split(",") if name.strip()]
            print(f"✓ Канонические имена сохранены: {names_str}")
        except Exception as e:
            print(f"Ошибка при сохранении канонических имен: {e}")

    def _save_aliases(self, aliases_str: str) -> bool:
        """Сохраняет алиасы в таблицу aliases"""
        try:
            # Проверяем валидность JSON
            aliases_dict = json.loads(aliases_str)

            conn = sqlite3.connect(DB_PROFILES)
            cursor = conn.cursor()

            # Для каждого алиаса в словаре находим user_id по canonical_name и сохраняем алиас
            for alias, canonical_name in aliases_dict.items():
                # Находим user_id по canonical_name
                cursor.execute('SELECT user_id FROM profiles WHERE canonical_name = ?', (canonical_name,))
                user_row = cursor.fetchone()
                if user_row:
                    user_id = user_row[0]
                    # Сохраняем алиас (INSERT OR IGNORE для избежания дубликатов)
                    cursor.execute('''
                        INSERT OR IGNORE INTO aliases (user_id, alias)
                        VALUES (?, ?)
                    ''', (user_id, alias))
                else:
                    print(f"Предупреждение: Каноническое имя '{canonical_name}' не найдено в profiles")

            conn.commit()
            conn.close()

            # Перезагружаем алиасы из базы данных
            self._load_names_from_db()
            print(f"✓ Алиасы сохранены в таблицу aliases: {aliases_str}")
            return True
        except json.JSONDecodeError:
            print(f"Ошибка: Неверный JSON формат для алиасов: {aliases_str}")
            return False
        except Exception as e:
            print(f"Ошибка при сохранении алиасов: {e}")
            return False

    def _is_message_from_allowed_user(self, canonical_name: str) -> bool:
        """Проверяет, является ли участник разрешенным для обработки сообщений"""
        return canonical_name in self.CANONICAL_NAMES

    def _get_user_id_from_canonical_name(self, canonical_name: str, from_id) -> int:
        """Преобразует каноническое имя и from_id в user_id (без префикса 'user')"""
        # Убираем префикс 'user' из from_id если он есть
        if str(from_id).startswith('user'):
            user_id = int(str(from_id)[4:])  # Убираем 'user' префикс
        else:
            user_id = int(from_id)
        return user_id

    def _save_message_embedding(self, user_id: int, message_text: str, message_id: Optional[str] = None, canonical_name: Optional[str] = None):
        """Сохраняет эмбеддинг для нового сообщения в ChromaDB"""
        try:
            # Получаем эмбеддинг для сообщения
            embedding = self._get_embedding_sync(message_text)
            if embedding is None:
                print(f"[EMBEDDING] Не удалось получить эмбеддинг для сообщения от {user_id}")
                return

            # Генерируем уникальный ID для документа
            import hashlib
            doc_id = hashlib.md5(f"{str(user_id)}_{message_text}_{time.time()}".encode()).hexdigest()

            # Сохраняем в ChromaDB
            if self.message_embeddings_collection:
                self.message_embeddings_collection.add(
                    documents=[message_text],
                    embeddings=[embedding.tolist()],  # Конвертируем numpy array в list
                    ids=[doc_id],
                    metadatas=[{
                        "user_id": user_id,
                        "message_text": message_text,
                        "timestamp": datetime.now().isoformat()
                    }]
                )
                display_name = f"{canonical_name} ({str(user_id)})" if canonical_name else str(user_id)
                print(f"[EMBEDDING] ✓ Сохранен эмбеддинг для нового сообщения от {display_name}: {message_text[:100]}...")
            else:
                print(f"[EMBEDDING] ✗ ChromaDB недоступна для сохранения эмбеддинга для {user_id}")
        except Exception as e:
            display_name = f"{canonical_name} ({user_id})" if canonical_name else user_id
            print(f"[EMBEDDING] ✗ Ошибка при сохранении эмбеддинга для {display_name}: {e}")


    def _is_message_already_processed(self, user_id: int, message_text: str) -> bool:
        """Проверяет, было ли уже обработано данное сообщение в ChromaDB"""
        try:
            if self.message_embeddings_collection:
                print(f"[DEBUG] Проверяем дубликат для user_id: {user_id}, сообщение: '{message_text[:50]}...'")
                # Поиск в ChromaDB по user_id и тексту сообщения
                results = self.message_embeddings_collection.get(
                    where={"$and": [{"user_id": user_id}, {"message_text": message_text}]},
                    limit=1
                )
                is_duplicate = len(results['ids']) > 0
                print(f"[DEBUG] Найден ли дублика сообщения: {is_duplicate}")
                return is_duplicate
        except Exception as e:
            print(f"Ошибка при проверке дубликата сообщения: {e}")
            return False

    def _should_save_embedding_for_message(self, update: Update, user_full_name: str, message_text: str) -> tuple[bool, Optional[int], Optional[str]]:
        """
        Проверяет, нужно ли сохранять эмбеддинг для сообщения.
        Возвращает (should_save, user_id, canonical_name) или (False, None, None) если не нужно.
        """
        # В личных чатах не создаем эмбеддинги для входящих сообщений, так как это обращения к боту
        if update.message.chat.type == "private":
            return False, None, None
        
        # В групповых чатах проверяем, является ли сообщение обращением к боту
        is_mentioned = self._is_bot_mentioned(update)
        if is_mentioned:
            # Это обращение к боту, не создаем эмбеддинг
            return False, None, None
        
        # Это обычное сообщение в группе, не для бота - проверяем, является ли отправитель разрешенным участником
        canonical_name = self._normalize_name(user_full_name)
        if not canonical_name or not self._is_message_from_allowed_user(canonical_name):
            # Отправитель не является разрешенным участником
            return False, None, None
        
        # Получаем user_id для сохранения в ChromaDB
        user_id = self._get_user_id_from_canonical_name(canonical_name, update.message.from_user.id)
        # user_id теперь возвращается как int
        
        # Проверяем, не обработано ли сообщение уже
        if self._is_message_already_processed(user_id, message_text):
            return False, None, None
        
        return True, user_id, canonical_name

    
    def _is_bot_mentioned(self, update: Update) -> bool:
        """
        Проверяет, упомянут ли бот в сообщении через @
        
        Args:
            update: Обновление от Telegram
            
        Returns:
            True если бот упомянут, False иначе
        """
        if not update.message:
            return False
        
        # В личных чатах всегда отвечаем (нет необходимости в упоминании)
        if update.message.chat.type == "private":
            return True
        
        # Используем сохраненный username бота
        bot_username = self.bot_username
        
        # Username должен быть получен через post_init, но на случай если он еще не получен,
        # мы не будем пытаться получить его здесь синхронно (это сложно из-за асинхронности)
        # Вместо этого просто проверяем, что он есть
        if not bot_username:
            # Это не должно происходить, так как username устанавливается автоматически при регистрации бота
            # и должен быть получен через post_init
            # Не отвечаем, пока username не получен
            return False
        
        # Проверяем entities на наличие упоминания бота (это единственный надежный способ)
        if update.message.entities:
            for entity in update.message.entities:
                if entity.type == "mention":
                    # Получаем текст упоминания
                    mention_text = update.message.text[entity.offset:entity.offset + entity.length].lower()
                    # Убираем @ для сравнения
                    mention_username = mention_text.lstrip("@")
                    if mention_username == bot_username:
                        return True
        
        # Если entities не содержат упоминания, бот не упомянут
        # print(f"[DEBUG] ✗ Бот не упомянут (username бота: @{bot_username})")
        return False
    
    async def handle_telegram_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает сообщения из Telegram"""
        if not update.message:
            return

        chat_id = update.message.chat.id
        chat_type = update.message.chat.type
        user = update.message.from_user
        user_id = user.id
        telegram_username = user.username
        username_display = telegram_username or user.first_name or "Unknown"
        user_full_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or username_display

        # Теперь проверяем, разрешен ли чат/группа/пользователь
        if not self._is_allowed_chat(chat_id):
            chat_type_str = "личный чат" if chat_id > 0 else "группа/чат"
            print(f"\n[Telegram] ⚠ Запрос из неразрешенного {chat_type_str} (ID: {chat_id})")
            print(f"[Telegram] Пользователь: {user_full_name} (@{username_display}, ID: {user_id})")
            print(f"[Telegram] Сообщение проигнорировано")
            if update.message.text:
                await update.message.reply_text("❌ Извините, этот бот доступен только для определенных чатов, групп и пользователей.")
            return

        # Обработка входящих сообщений для создания эмбеддингов (до проверки упоминания бота)
        # Создаем эмбеддинги только для сообщений НЕ к боту (без упоминания в группах и не в личных сообщениях)
        if update.message.text:
            message_text = update.message.text.strip()

            # Нормализуем имя отправителя для определения канонического имени
            from_name = user_full_name
            canonical_name = self._normalize_name(from_name)

            # Проверяем, упоминает ли сообщение бота
            is_mentioned = self._is_bot_mentioned(update) if chat_type != "private" else False

            # Если отправитель - разрешенный участник и сообщение НЕ для бота, сохраняем эмбеддинг
            if canonical_name and self._is_message_from_allowed_user(canonical_name) and not is_mentioned:
                # Получаем user_id для сохранения в ChromaDB
                user_id_clean = self._get_user_id_from_canonical_name(canonical_name, user_id)
                # user_id_clean теперь int

                # Создаем алиасы для пользователя при первом обращении
                if user_id_clean:
                    # Собираем все возможные алиасы: first_name, last_name, username, full_name
                    user_aliases = []
                    if user.first_name:
                        user_aliases.append(user.first_name)
                    if user.last_name:
                        user_aliases.append(user.last_name)
                    if user.username:
                        user_aliases.append(user.username)
                    if user_full_name and user_full_name != user.username:
                        user_aliases.append(user_full_name)

                    # Убираем дубликаты
                    user_aliases = list(set(user_aliases))

                    # Проверяем, есть ли уже алиасы для этого пользователя
                    existing_aliases = self._get_user_aliases(int(user_id_clean))
                    if not existing_aliases:
                        # Сохраняем алиасы при первом обращении
                        self._save_user_aliases(int(user_id_clean), user_aliases)
                    else:
                        # Проверяем, есть ли новые алиасы для добавления
                        new_aliases = [alias for alias in user_aliases if alias not in existing_aliases]
                        if new_aliases:
                            self._save_user_aliases(int(user_id_clean), new_aliases)

                # Асинхронно сохраняем эмбеддинг для нового сообщения
                try:
                    # Запускаем сохранение эмбеддинга в фоне, чтобы не блокировать обработку
                    asyncio.create_task(self._save_message_embedding_async(user_id_clean, message_text, canonical_name=canonical_name))
                except Exception as e:
                    print(f"[EMBEDDING] Ошибка при запуске сохранения эмбеддинга: {e}")

        # В групповых чатах проверяем упоминание бота
        if chat_type != "private" and update.message.text:
            is_mentioned = self._is_bot_mentioned(update)
            if not is_mentioned:
                # Сообщение не для бота, выходим (эмбеддинг уже сохранен выше)
                return
            # Бот упомянут - продолжаем обработку для ответа боту

        # Если нет текста, выходим
        if not update.message.text:
            return

        message_text = update.message.text.strip()

        # В групповых чатах проверяем упоминание бота
        if chat_type != "private" and update.message.text:
            is_mentioned = self._is_bot_mentioned(update)
            if not is_mentioned:
                # Сообщение не для бота, выходим (эмбеддинг уже сохранен выше)
                return
            # Бот упомянут - продолжаем обработку для ответа боту

        # Если нет текста, выходим
        if not update.message.text:
            return

        message_text = update.message.text.strip()

        chat = update.message.chat
        if chat.type == "private":
            chat_name = user_full_name or chat.username or f"User {chat_id}"
        else:
            chat_name = chat.title or chat.username or f"Chat {chat_id}"

        mention_name = telegram_username if telegram_username else f"id{user_id}"
        sanitized_message = " ".join(message_text.split()) or message_text

        # Логируем обращение пользователя в требуемом формате
        print(f"[@{mention_name}] chat: {chat_name} ({chat_id}). {sanitized_message}")
        print(f"[Telegram] Начинаю обработку сообщения...")

        try:
            # Парсим вопрос
            parsed_result = self.parse_question(message_text)
            if len(parsed_result) == 3:
                question, name, verbose = parsed_result
            else:
                # Обратная совместимость со старым форматом
                question, name = parsed_result
                verbose = False

            if not name:
                error_msg = "Ошибка: Укажите имя участника в формате ^Имя^\n\nДоступные участники:\n"
                for canonical_name in self.messages_by_user.keys():
                    error_msg += f"  • {canonical_name}\n"
                await update.message.reply_text(error_msg)
                print(f"[Telegram] Ответ: {error_msg}")
                return

            # Находим участника
            canonical_name = self.find_user_by_name(name)

            if not canonical_name:
                error_msg = f"Ошибка: Участник '{name}' не найден\n\nДоступные участники:\n"
                for available_name in self.messages_by_user.keys():
                    error_msg += f"  • {available_name}\n"
                await update.message.reply_text(error_msg)
                print(f"[Telegram] Ответ: {error_msg}")
                return

            # Отправляем сообщение о том, что обрабатываем
            processing_msg = await update.message.reply_text(f"🤔 [{canonical_name} думает...]" + (" (verbose)" if verbose else ""))

            print(f"[Telegram] Генерация ответа в стиле {canonical_name}...")

            # Генерируем ответ
            response = await self.generate_response(question, canonical_name, verbose=verbose)

            # Логируем ответ в консоль
            print(f"[Telegram] {canonical_name}: {response}")

            # Отправляем ответ в Telegram (используем HTML для более надежного форматирования)
            formatted_response = f"💬 <b>{canonical_name}:</b>\n\n{response}"
            await processing_msg.edit_text(formatted_response, parse_mode='HTML')

        except Exception as e:
            error_msg = f"Произошла ошибка: {str(e)}"
            await update.message.reply_text(error_msg)
            print(f"[Telegram] Ошибка: {error_msg}")

    async def _save_message_embedding_async(self, user_id: int, message_text: str, canonical_name: Optional[str] = None):
        """Асинхронно сохраняет эмбеддинг для нового сообщения"""
        try:
            await asyncio.get_event_loop().run_in_executor(None, self._save_message_embedding, user_id, message_text, None, canonical_name)
        except Exception as e:
            print(f"[EMBEDDING] Ошибка при асинхронном сохранении эмбеддинга: {e}")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает команду /start"""
        chat_id = update.message.chat.id
        
        # Проверяем, разрешен ли чат/группа/пользователь
        if not self._is_allowed_chat(chat_id):
            await update.message.reply_text("❌ Извините, этот бот доступен только для определенных чатов, групп и пользователей.")
            return
        
        help_text = """👋 Привет! Я бот, который отвечает в стиле участников группы.

📝 Формат вопросов:
Какая музыка тебе нравится ^Тимур^
Что думаешь о погоде ^Юрий^

Доступные участники:

... кому надо, тот знает ...
"""
        # for canonical_name in self.messages_by_user.keys():
            # help_text += f"  • {canonical_name}\n"
        
        await update.message.reply_text(help_text)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает команду /help"""
        chat_id = update.message.chat.id
        
        # Проверяем, разрешен ли чат/группа/пользователь
        if not self._is_allowed_chat(chat_id):
            await update.message.reply_text("❌ Извините, этот бот доступен только для определенных чатов, групп и пользователей.")
            return
        
        await self.start_command(update, context)
    
    async def list_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает команду /list - показывает список участников"""
        chat_id = update.message.chat.id

        # Проверяем, разрешен ли чат/группа/пользователь
        if not self._is_allowed_chat(chat_id):
            await update.message.reply_text("❌ Извините, этот бот доступен только для определенных чатов, групп и пользователей.")
            return

        if not self.messages_by_user:
            await update.message.reply_text("Участники еще не загружены.")
            return

        list_text = "📋 Доступные участники:\n\n"
        for canonical_name in self.messages_by_user.keys():
            count = len(self.messages_by_user[canonical_name])
            list_text += f"  • {canonical_name} ({count} сообщений)\n"

        await update.message.reply_text(list_text)

    async def admin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает команду /admin - показывает административные команды"""
        chat_id = update.message.chat.id

        # Проверяем, разрешен ли чат/группа/пользователь
        if not self._is_allowed_chat(chat_id):
            await update.message.reply_text("❌ Извините, этот бот доступен только для определенных чатов, групп и пользователей.")
            return

        admin_text = """🔧 Административные команды:

   /set_canonical_names <список> - Установить канонические имена (через запятую)
   /get_canonical_names - Показать текущие канонические имена
   /set_name_aliases <json> - Установить маппинг алиасов (JSON формат, сохраняется в таблицу aliases)
   /get_name_aliases - Показать текущий маппинг алиасов (из таблицы aliases)
   /set_allowed_chat_ids <список> - Установить разрешенные ID чатов (через запятую)
   /get_allowed_chat_ids - Показать текущие разрешенные ID чатов
   /set_ollama_model <модель> - Установить модель Ollama
   /get_ollama_model - Показать текущую модель Ollama
   /get_ollama_models - Показать список локальных моделей Ollama
   /update_embeddings <json> - Обновить эмбеддинги из JSON файла с диалогами

   Пример:
   • /set_canonical_names Иван,Петр,Сидор
   • /set_name_aliases {"Ваня":"Иван","Петя":"Петр"}
   • /set_allowed_chat_ids 123456789,-1001234567890
   • /set_ollama_model yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest
   • /update_embeddings - отправьте JSON файл с диалогами"""

        await update.message.reply_text(admin_text)

    async def set_canonical_names_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает команду /set_canonical_names - устанавливает канонические имена"""
        chat_id = update.message.chat.id

        # Проверяем, разрешен ли чат/группа/пользователь
        if not self._is_allowed_chat(chat_id):
            await update.message.reply_text("❌ Извините, этот бот доступен только для определенных чатов, групп и пользователей.")
            return

        # Получаем аргументы команды
        args = context.args
        if not args:
            await update.message.reply_text("❌ Укажите канонические имена через запятую\nПример: /set_canonical_names Иван,Петр,Сидор")
            return

        names_str = " ".join(args)
        self._save_canonical_names(names_str)
        await update.message.reply_text(f"✅ Канонические имена установлены: {names_str}")

    async def get_canonical_names_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает команду /get_canonical_names - показывает канонические имена"""
        chat_id = update.message.chat.id

        # Проверяем, разрешен ли чат/группа/пользователь
        if not self._is_allowed_chat(chat_id):
            await update.message.reply_text("❌ Извините, этот бот доступен только для определенных чатов, групп и пользователей.")
            return

        if self.CANONICAL_NAMES:
            names_text = "📋 Канонические имена:\n\n" + "\n".join(f"• {name}" for name in self.CANONICAL_NAMES)
        else:
            names_text = "📋 Канонические имена не установлены"

        await update.message.reply_text(names_text)

    async def set_aliases_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает команду /set_aliases - устанавливает алиасы"""
        chat_id = update.message.chat.id

        # Проверяем, разрешен ли чат/группа/пользователь
        if not self._is_allowed_chat(chat_id):
            await update.message.reply_text("❌ Извините, этот бот доступен только для определенных чатов, групп и пользователей.")
            return

        # Получаем аргументы команды
        args = context.args
        if not args:
            await update.message.reply_text("❌ Укажите алиасы в формате JSON\nПример: /set_aliases {\"Ваня\":\"Иван\",\"Петя\":\"Петр\"}")
            return

        aliases_str = " ".join(args)
        if self._save_aliases(aliases_str):
            await update.message.reply_text(f"✅ Алиасы установлены: {aliases_str}")
        else:
            await update.message.reply_text("❌ Ошибка: Неверный JSON формат алиасов")

    async def get_aliases_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает команду /get_name_aliases - показывает маппинг имен"""
        chat_id = update.message.chat.id

        # Проверяем, разрешен ли чат/группа/пользователь
        if not self._is_allowed_chat(chat_id):
            await update.message.reply_text("❌ Извините, этот бот доступен только для определенных чатов, групп и пользователей.")
            return

        if self.NAME_ALIASES:
            import json
            aliases_text = "📋 Алиасы имен:\n\n" + json.dumps(self.NAME_ALIASES, indent=2, ensure_ascii=False)
        else:
            aliases_text = "📋 Алиасы имен не установлен"

        await update.message.reply_text(aliases_text)

    async def set_allowed_chat_ids_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает команду /set_allowed_chat_ids - устанавливает разрешенные ID чатов"""
        chat_id = update.message.chat.id
        user_id = update.message.from_user.id

        # Проверяем, является ли пользователь администратором
        if user_id != self.admin_user_id:
            await update.message.reply_text("❌ У вас нет прав для выполнения этой команды")
            return

        # Проверяем, разрешен ли чат/группа/пользователь
        if not self._is_allowed_chat(chat_id):
            await update.message.reply_text("❌ Извините, этот бот доступен только для определенных чатов, групп и пользователей.")
            return

        # Получаем аргументы команды
        args = context.args
        if not args:
            await update.message.reply_text("❌ Укажите разрешенные ID чатов через запятую\nПример: /set_allowed_chat_ids 123456789,-1001234567890\n\nОставьте пустым для разрешения всех чатов")
            return

        chat_ids_str = " ".join(args)
        try:
            # Парсим и валидируем ID чатов
            if chat_ids_str.strip():
                allowed_chat_ids = [int(chat_id.strip()) for chat_id in chat_ids_str.split(",") if chat_id.strip()]
                self._save_allowed_chat_ids(allowed_chat_ids)
                await update.message.reply_text(f"✅ Разрешенные ID чатов установлены: {chat_ids_str}")
            else:
                # Пустая строка - разрешаем все чаты
                self._save_allowed_chat_ids([])
                await update.message.reply_text("✅ Разрешенные ID чатов сброшены - бот будет принимать сообщения из всех чатов")
        except ValueError:
            await update.message.reply_text("❌ Ошибка: ID чатов должны быть целыми числами\nПример: /set_allowed_chat_ids 123456789,-1001234567890")

    async def get_allowed_chat_ids_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает команду /get_allowed_chat_ids - показывает разрешенные ID чатов"""
        chat_id = update.message.chat.id
        user_id = update.message.from_user.id

        # Проверяем, является ли пользователь администратором
        if user_id != self.admin_user_id:
            await update.message.reply_text("❌ У вас нет прав для выполнения этой команды")
            return

        # Проверяем, разрешен ли чат/группа/пользователь
        if not self._is_allowed_chat(chat_id):
            await update.message.reply_text("❌ Извините, этот бот доступен только для определенных чатов, групп и пользователей.")
            return

        allowed_chat_ids = self._load_allowed_chat_ids()
        if allowed_chat_ids:
            ids_text = "📋 Разрешенные ID чатов:\n\n" + "\n".join(f"• {chat_id}" for chat_id in allowed_chat_ids)
        else:
            ids_text = "📋 Разрешенные ID чатов не установлены (бот принимает сообщения из всех чатов)"

        await update.message.reply_text(ids_text)

    async def set_ollama_model_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает команду /set_ollama_model - устанавливает модель Ollama"""
        chat_id = update.message.chat.id
        user_id = update.message.from_user.id

        # Проверяем, является ли пользователь администратором
        if user_id != self.admin_user_id:
            await update.message.reply_text("❌ У вас нет прав для выполнения этой команды")
            return

        # Проверяем, разрешен ли чат/группа/пользователь
        if not self._is_allowed_chat(chat_id):
            await update.message.reply_text("❌ Извините, этот бот доступен только для определенных чатов, групп и пользователей.")
            return

        # Получаем аргументы команды
        args = context.args
        if not args:
            await update.message.reply_text("❌ Укажите модель Ollama\nПример: /set_ollama_model yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest")
            return

        model_name = " ".join(args)
        self._save_ollama_model(model_name)
        await update.message.reply_text(f"✅ Модель Ollama установлена: {model_name}")

    async def get_ollama_model_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает команду /get_ollama_model - показывает текущую модель Ollama"""
        chat_id = update.message.chat.id
        user_id = update.message.from_user.id

        # Проверяем, является ли пользователь администратором
        if user_id != self.admin_user_id:
            await update.message.reply_text("❌ У вас нет прав для выполнения этой команды")
            return

        # Проверяем, разрешен ли чат/группа/пользователь
        if not self._is_allowed_chat(chat_id):
            await update.message.reply_text("❌ Извините, этот бот доступен только для определенных чатов, групп и пользователей.")
            return

        current_model = self._load_ollama_model()
        await update.message.reply_text(f"🤖 Текущая модель Ollama: {current_model}")

    async def update_embeddings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает команду /update_embeddings - обновляет эмбеддинги из JSON файла"""
        chat_id = update.message.chat.id
        user_id = update.message.from_user.id

        # Проверяем, является ли пользователь администратором
        if user_id != self.admin_user_id:
            await update.message.reply_text("❌ У вас нет прав для выполнения этой команды")
            return

        # Проверяем, разрешен ли чат/группа/пользователь
        if not self._is_allowed_chat(chat_id):
            await update.message.reply_text("❌ Извините, этот бот доступен только для определенных чатов, групп и пользователей.")
            return

        await update.message.reply_text("📎 Пожалуйста, отправьте JSON файл с диалогами для обновления эмбеддингов")

    async def get_ollama_models_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает команду /get_ollama_models - показывает список локальных моделей Ollama"""
        chat_id = update.message.chat.id
        user_id = update.message.from_user.id

        # Проверяем, является ли пользователь администратором
        if user_id != self.admin_user_id:
            await update.message.reply_text("❌ У вас нет прав для выполнения этой команды")
            return

        # Проверяем, разрешен ли чат/группа/пользователь
        if not self._is_allowed_chat(chat_id):
            await update.message.reply_text("❌ Извините, этот бот доступен только для определенных чатов, групп и пользователей.")
            return

        # Получаем список моделей в фоне
        processing_msg = await update.message.reply_text("🤖 Получаю список моделей Ollama...")

        try:
            # Выполняем запрос в отдельном потоке
            loop = asyncio.get_event_loop()
            models = await loop.run_in_executor(None, self._get_ollama_models_sync)

            if models:
                models_text = "🤖 Локальные модели Ollama:\n\n" + "\n".join(f"• {model}" for model in sorted(models))
            else:
                models_text = "🤖 Не удалось получить список моделей Ollama или модели не установлены"

            await processing_msg.edit_text(models_text)
        except Exception as e:
            error_text = f"❌ Ошибка при получении списка моделей: {str(e)}"
            await processing_msg.edit_text(error_text)

    async def _handle_document_admin(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает документы от администратора для обновления эмбеддингов"""
        if not update.message.document:
            return

        user_id = update.message.from_user.id

        # Проверяем, является ли пользователь администратором
        if user_id != self.admin_user_id:
            return

        chat_id = update.message.chat.id
        if not self._is_allowed_chat(chat_id):
            return

        try:
            # Скачиваем файл
            file = await update.message.document.get_file()
            file_content = await file.download_as_bytearray()
            json_data = json.loads(file_content.decode('utf-8'))

            await update.message.reply_text("📊 Начинаю обработку файла и создание эмбеддингов...")

            # Обрабатываем JSON и создаем эмбеддинги
            processed_count = await self._process_embeddings_from_json(json_data)

            await update.message.reply_text(f"✅ Эмбеддинги успешно обновлены! Обработано {processed_count} сообщений")

        except json.JSONDecodeError:
            await update.message.reply_text("❌ Ошибка: Файл не является корректным JSON")
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка при обработке файла: {str(e)}")

    async def _process_embeddings_from_json(self, json_data: dict) -> int:
        """Обрабатывает JSON данные и создает эмбеддинги для пользователей"""
        total_processed = 0

        try:
            messages = json_data.get("messages", [])
            print(f"[ADMIN] Получено {len(messages)} сообщений из JSON файла")

            # Группируем сообщения по пользователям
            user_messages = defaultdict(list)

            for msg in messages:
                # Пропускаем служебные сообщения
                if msg.get("type") != "message":
                    continue

                from_name = msg.get("from")
                from_id = msg.get("from_id")

                # Нормализуем имя к каноническому варианту
                canonical_name = self._normalize_name(from_name)
                if not canonical_name:
                    continue

                # Обрабатываем text (может быть строкой или массивом)
                text_raw = msg.get("text", "")
                if isinstance(text_raw, list):
                    # Если text - массив, извлекаем только plain текст
                    text_parts = []
                    for item in text_raw:
                        if isinstance(item, dict):
                            if item.get("type") == "plain":
                                text_parts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            # Если элемент - строка, добавляем её
                            text_parts.append(item)
                    text = " ".join(text_parts).strip()
                else:
                    text = str(text_raw).strip() if text_raw else ""

                # Извлекаем только plain текст из text_entities
                text_entities = msg.get("text_entities", [])
                plain_text_parts = []

                for entity in text_entities:
                    if entity.get("type") == "plain":
                        plain_text_parts.append(entity.get("text", ""))

                # Если есть plain текст из entities, используем его, иначе используем text
                if plain_text_parts:
                    final_text = " ".join(plain_text_parts).strip()
                elif text:
                    final_text = text
                else:
                    # Нет текстового содержимого, пропускаем
                    continue

                # Пропускаем если текст пустой или слишком короткий
                if not final_text or len(final_text) < 2:
                    continue

                # Получаем user_id для сохранения в ChromaDB
                user_id_clean = self._get_user_id_from_canonical_name(canonical_name, from_id)

                # Добавляем сообщение к пользователю
                user_messages[user_id_clean].append(final_text)
                total_processed += 1

            # Создаем эмбеддинги для каждого пользователя
            for user_id, messages_list in user_messages.items():
                await self._create_embeddings_from_messages(user_id, messages_list)

            print(f"[ADMIN] Обработано {total_processed} сообщений для {len(user_messages)} пользователей")

        except Exception as e:
            print(f"[ADMIN] Ошибка при обработке JSON данных: {e}")
            raise e

        return total_processed

    async def _create_embeddings_from_messages(self, user_id: int, messages: List[str]):
        """Создает эмбеддинги для списка сообщений пользователя"""
        try:
            print(f"[ADMIN] Создание эмбеддингов для user_id {user_id} ({len(messages)} сообщений)")

            for i, message in enumerate(messages):
                if (i + 1) % 50 == 0:
                    print(f"[ADMIN]    Обработано {i + 1}/{len(messages)} сообщений...")

                # Проверяем, не обработано ли сообщение уже
                if self._is_message_already_processed(user_id, message):
                    continue

                # Создаем эмбеддинг
                embedding = self._get_embedding_sync(message)
                if embedding is not None:
                    # Генерируем уникальный ID для документа
                    import hashlib
                    doc_id = hashlib.md5(f"{str(user_id)}_{message}_{time.time()}".encode()).hexdigest()

                    # Сохраняем в ChromaDB
                    if self.message_embeddings_collection:
                        self.message_embeddings_collection.add(
                            documents=[message],
                            embeddings=[embedding.tolist()],
                            ids=[doc_id],
                            metadatas=[{
                                "user_id": str(user_id),
                                "message_text": message,
                                "timestamp": datetime.now().isoformat(),
                                "source": "admin_upload"
                            }]
                        )
                        print(f"[ADMIN] ✓ Сохранен эмбеддинг для сообщения от user_id {user_id}")

        except Exception as e:
            print(f"[ADMIN] ✗ Ошибка при создании эмбеддингов для user_id {user_id}: {e}")

    async def post_init(self, application: Application):
        """Вызывается после инициализации бота"""
        # Сохраняем username бота для проверки упоминаний
        try:
            bot_info = await application.bot.get_me()
            self.bot_username = bot_info.username.lower() if bot_info.username else None
            print(f"✓ Telegram бот инициализирован и готов к работе")
            if self.bot_username:
                print(f"✓ Username бота: @{bot_info.username}")
            else:
                print(f"⚠ Внимание: Username бота не установлен! Бот не сможет отвечать в групповых чатах.")
                print(f"   Установите username через @BotFather командой /setusername")

            # Инициализируем начальные настройки из .env в базу данных (если они еще не загружены)
            self._init_settings_from_env()
            # Загружаем настройки из базы данных
            self._load_allowed_chat_ids()
            self._load_ollama_model()
            self._load_ollama_embedding_model()

            # Загружаем существующие профили (если они есть), чтобы получить user_id для обновления алиасов
            self.load_profiles()

            # Обновляем алиасы всех известных пользователей из Telegram API
            print("перед вызовом алиасов")
            await self._update_all_user_aliases_from_telegram(application.bot)
        except Exception as e:
            print(f"⚠ Не удалось получить username бота: {e}")
            print("✓ Telegram бот инициализирован и готов к работе")
    
    def setup_telegram_bot(self, token: str):
        """Настраивает Telegram бота"""
        try:
            self.telegram_app = Application.builder().token(token).build()

            # Регистрируем обработчики
            self.telegram_app.add_handler(CommandHandler("start", self.start_command))
            self.telegram_app.add_handler(CommandHandler("help", self.help_command))
            self.telegram_app.add_handler(CommandHandler("list", self.list_command))
            self.telegram_app.add_handler(CommandHandler("admin", self.admin_command))
            self.telegram_app.add_handler(CommandHandler("set_canonical_names", self.set_canonical_names_command))
            self.telegram_app.add_handler(CommandHandler("get_canonical_names", self.get_canonical_names_command))
            self.telegram_app.add_handler(CommandHandler("set_aliases", self.set_aliases_command))
            self.telegram_app.add_handler(CommandHandler("get_name_aliases", self.get_aliases_command))
            self.telegram_app.add_handler(CommandHandler("set_allowed_chat_ids", self.set_allowed_chat_ids_command))
            self.telegram_app.add_handler(CommandHandler("get_allowed_chat_ids", self.get_allowed_chat_ids_command))
            self.telegram_app.add_handler(CommandHandler("get_ollama_model", self.get_ollama_model_command))
            self.telegram_app.add_handler(CommandHandler("set_ollama_model", self.set_ollama_model_command))
            self.telegram_app.add_handler(CommandHandler("update_embeddings", self.update_embeddings_command))
            self.telegram_app.add_handler(CommandHandler("get_ollama_models", self.get_ollama_models_command))
            
            # Обрабатываем документы от администратора
            self.telegram_app.add_handler(MessageHandler(filters.Document.ALL, self._handle_document_admin))
            # Обрабатываем текстовые сообщения (включая упоминания в группах)
            self.telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_telegram_message))

            print("✓ Telegram бот настроен")

            return True
        except Exception as e:
            print(f"✗ Ошибка при настройке Telegram бота: {e}")
            return False
    
    async def run_telegram_bot(self):
        """Запускает Telegram бота"""
        if not self.telegram_app:
            print("✗ Telegram бот не настроен")
            return
        
        print("🚀 Запуск Telegram бота...")
        await self.telegram_app.initialize()
        await self.telegram_app.start()

        # Обновляем алиасы всех пользователей из Telegram API
        await self._update_all_user_aliases_from_telegram(self.telegram_app.bot)

        # Убеждаемся, что username получен перед началом обработки сообщений
        # post_init должен был уже выполниться, но на всякий случай проверяем
        if not self.bot_username:
            try:
                print("Получение username бота...")
                bot_info = await self.telegram_app.bot.get_me()
                self.bot_username = bot_info.username.lower() if bot_info.username else None
                if self.bot_username:
                    print(f"✓ Username бота получен: @{bot_info.username}")
                else:
                    print(f"⚠ Внимание: Username бота не установлен в Telegram!")
                    print(f"   Установите username через @BotFather командой /setusername")
            except Exception as e:
                print(f"⚠ Ошибка при получении username бота: {e}")
                import traceback
                traceback.print_exc()
        
        await self.telegram_app.updater.start_polling()
        print("✓ Telegram бот запущен и слушает сообщения")
    
    def run(self, force_recreate: bool = False, telegram_mode: bool = False, rebuild_profiles_only: bool = False, rebuild_embeddings: bool = False):
        """Запускает бота"""
        print("=" * 60)
        print("Telegram Bot - Генератор ответов в стиле участников")
        print("=" * 60)

        # Загружаем сообщения, только если указан флаг или это первый запуск
        if load_messages or not self.messages_by_user:
            self.load_messages()

        # Создаем портреты (или загружаем сохраненные)
        # Сначала пытаемся загрузить существующие профили
        if not force_recreate and not rebuild_profiles_only:
            profiles_loaded = self.load_profiles()
            if profiles_loaded:
                print("✓ Используются сохраненные портреты участников (быстрая загрузка)")
                if rebuild_embeddings:
                    # Принудительно пересоздаем эмбеддинги если передан флаг rebuild_embeddings
                    self.create_all_profiles(force_recreate=False, rebuild_embeddings=True)
                    print("ℹ Режим пересоздания эмбеддингов активирован")
                # Выводим статистику эмбеддингов
                self._print_embeddings_stats()
                return

        # Если профили не загружены или требуется пересоздание
        profiles_force = force_recreate or rebuild_profiles_only
        if rebuild_embeddings:
            # Принудительно пересоздаем эмбеддинги если передан флаг rebuild_embeddings
            self.create_all_profiles(force_recreate=profiles_force, rebuild_embeddings=True)
            print("ℹ Режим пересоздания эмбеддингов активирован")
        else:
            rebuild_embeddings_flag = not rebuild_profiles_only
            self.create_all_profiles(force_recreate=profiles_force, rebuild_embeddings=rebuild_embeddings_flag)
            if rebuild_profiles_only:
                print("ℹ Режим пересоздания только профилей активирован (эмбеддинги сохранены)")

        # Выводим статистику эмбеддингов
        self._print_embeddings_stats()

        # Если режим Telegram, запускаем Telegram бота
        if telegram_mode:
            asyncio.run(self._run_telegram_mode_async(force_recreate, rebuild_profiles_only))
        else:
            # Режим командной строки
            self._run_console_mode()

    async def _run_telegram_mode_async(self, force_recreate: bool = False, rebuild_profiles_only: bool = False):
        """Запускает бота в режиме Telegram с асинхронной инициализацией"""
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            print("✗ Ошибка: TELEGRAM_BOT_TOKEN не найден в переменных окружения")
            print("Создайте файл .env и добавьте: TELEGRAM_BOT_TOKEN=your_token_here")
            return

        # Проверяем настройку разрешенных чатов/групп/пользователей
        allowed_chat_ids = self._load_allowed_chat_ids()
        if allowed_chat_ids:
            print(f"✓ Бот будет принимать сообщения только из разрешенных чатов/групп/пользователей (ID: {', '.join(str(chat_id) for chat_id in allowed_chat_ids)})")
        else:
            print("ℹ Разрешенные ID чатов не заданы - бот будет принимать сообщения из всех чатов, групп и пользователей")
            print("  Для ограничения доступа используйте команду /set_allowed_chat_ids")

        if not self.setup_telegram_bot(token):
            return

        print("\n" + "=" * 60)
        print("Бот готов! Работает в режиме Telegram")
        print("Также доступен режим командной строки (запустите без --telegram)")
        print("=" * 60 + "\n")

        # Запускаем бота сразу, не дожидаясь завершения создания эмбеддингов
        try:
            await self.run_telegram_bot()
            # Держим бота запущенным
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            print("\n\nОстановка Telegram бота...")
            await self.telegram_app.stop()
            await self.telegram_app.shutdown()
            print("До свидания!")


    def run(self, load_messages: bool = False, update_profiles: bool = False, update_embeddings: bool = False, telegram_mode: bool = False):
        """Запускает бота"""
        print("=" * 60)
        print("Telegram Bot - Генератор ответов в стиле участников")
        print("=" * 60)

        # В режиме Telegram всегда загружаем сообщения, только если они не загружены
        if telegram_mode and not self.messages_by_user:
            load_messages = True
            # Не устанавливаем update_profiles=True, чтобы не пересоздавать профили каждый раз
            # update_embeddings сохраняем как есть, если был указан флаг --update-embeddings

        # Загружаем сообщения только если указан флаг
        if load_messages and os.path.exists(self.json_file):
            self.load_messages()

        # Создаем профили, если они не загружены (в Telegram режиме всегда пытаемся загрузить существующие)
        if telegram_mode or update_profiles:
            force_recreate = update_profiles  # Принудительно пересоздаем только если явно указано
            self.create_all_profiles(force_recreate=force_recreate, rebuild_embeddings=update_embeddings)

        # # Обновляем эмбеддинги только если указан флаг (без пересоздания профилей)
        # if update_embeddings:
        #     # Загружаем сообщения, если они еще не загружены
        #     if not self.messages_by_user:
        #         self.load_messages()
        #     self._create_all_embeddings_chromadb(force_recreate=True)

        # Выводим статистику эмбеддингов
        self._print_embeddings_stats()

        # Если режим Telegram, запускаем Telegram бота
        if telegram_mode:
            asyncio.run(self._run_telegram_mode_async())
        else:
            # Режим командной строки
            self._run_console_mode()


if __name__ == "__main__":
    # Загружаем переменные окружения из .env файла
    load_dotenv()

    # Проверяем аргументы командной строки
    load_messages = "--load-messages" in sys.argv or "--load" in sys.argv or "-l" in sys.argv
    update_profiles = "--update-profiles" in sys.argv or "--profiles" in sys.argv or "-p" in sys.argv
    update_embeddings = "--update-embeddings" in sys.argv or "--embeddings" in sys.argv or "-e" in sys.argv
    telegram_mode = "--telegram" in sys.argv or "-t" in sys.argv

    # Парсим --messages-file
    messages_file = "result.json"  # значение по умолчанию
    if "--messages-file" in sys.argv or "--file" in sys.argv or "-f" in sys.argv:
        file_flag = None
        for flag in ["--messages-file", "--file", "-f"]:
            if flag in sys.argv:
                file_flag = flag
                break

        if file_flag:
            try:
                file_index = sys.argv.index(file_flag)
                if file_index + 1 < len(sys.argv):
                    messages_file = sys.argv[file_index + 1]
                else:
                    print("✗ Ошибка: После {} не указано имя файла".format(file_flag))
                    sys.exit(1)
            except ValueError:
                pass

    # Создаем бота с указанным файлом сообщений
    bot = TelegramBot(json_file=messages_file)
    bot.run(load_messages=load_messages, update_profiles=update_profiles, update_embeddings=update_embeddings, telegram_mode=telegram_mode)

