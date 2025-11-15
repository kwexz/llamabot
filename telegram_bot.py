import json
import re
import requests
import os
import sys
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

# Настройки Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest"
DB_FILE = "user_profiles.db"
PROFILES_VERSION = "1.0"  # Версия формата профилей

class TelegramBot:
    # Канонические имена участников (основные имена для профилей)
    CANONICAL_NAMES = [
        "Юрий Курков",
        "Тимур Саркаров",
        "Игорь Толстов",
        "Илья Самойлов",
        "Степан Горгуца"
    ]
    
    # Маппинг вариантов написания имен на канонические имена
    # (для объединения сообщений одного человека с разными вариантами имени)
    NAME_MAPPING = {
        "Ilya Samoylov": "Илья Самойлов",
        "Илья Самойлов": "Илья Самойлов"
    }
    
    def __init__(self, json_file: str = "result.json"):
        self.json_file = json_file
        self.messages_by_user: Dict[str, List[str]] = defaultdict(list)  # canonical_name -> messages
        self.user_profiles: Dict[str, str] = {}  # canonical_name -> profile
        self.user_names: Dict[str, str] = {}  # from_id -> canonical_name
        self.name_variants: Dict[str, List[str]] = defaultdict(list)  # canonical_name -> [variants]
        # Инициализируем базу данных
        self._init_database()
    
    def _normalize_name(self, name: str) -> Optional[str]:
        """Нормализует имя пользователя к каноническому варианту"""
        if not name:
            return None
        
        # Проверяем маппинг
        if name in self.NAME_MAPPING:
            return self.NAME_MAPPING[name]
        
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
    
    def _is_allowed_user(self, name: str) -> bool:
        """Проверяет, является ли пользователь разрешенным"""
        return self._normalize_name(name) is not None
    
    def _init_database(self):
        """Инициализирует базу данных SQLite"""
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            
            # Создаем таблицу для профилей
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS profiles (
                    canonical_name TEXT PRIMARY KEY,
                    profile_text TEXT NOT NULL,
                    training_date TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Создаем таблицу для метаданных
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Ошибка при инициализации базы данных: {e}")
        
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
        
        print(f"Обработано сообщений для {len(self.messages_by_user)} участников:")
        for canonical_name, messages in self.messages_by_user.items():
            count = len(messages)
            variants = self.name_variants.get(canonical_name, [])
            if len(variants) > 1:
                print(f"  {canonical_name} (варианты: {', '.join(variants)}): {count} сообщений")
            else:
                print(f"  {canonical_name}: {count} сообщений")
    
    def call_ollama(self, prompt: str) -> str:
        """Вызывает Ollama API для генерации ответа"""
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            return f"Ошибка при обращении к Ollama: {e}"
    
    def create_profile(self, canonical_name: str) -> str:
        """Создает портрет участника на основе его сообщений"""
        messages = self.messages_by_user[canonical_name]
        
        if not messages:
            return ""
        
        # Берем последние 50 сообщений для анализа (чтобы не перегружать промпт)
        recent_messages = messages[-50:] if len(messages) > 50 else messages
        
        # Формируем примеры сообщений
        examples = "\n".join([f"- {msg}" for msg in recent_messages[:30]])
        
        prompt = f"""Проанализируй стиль общения этого человека на основе его сообщений и создай краткий портрет его стиля общения.

Имя: {canonical_name}
Примеры сообщений:
{examples}

Создай краткое описание стиля общения этого человека (2-3 предложения), включая:
- Типичные фразы и выражения
- Стиль речи (формальный/неформальный, краткий/развернутый)
- Особенности лексики
- Тон общения

Портрет стиля:"""
        
        profile = self.call_ollama(prompt)
        return profile
    
    def load_profiles(self) -> bool:
        """Загружает сохраненные портреты из базы данных"""
        if not os.path.exists(DB_FILE):
            return False
        
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            
            # Загружаем профили
            cursor.execute('SELECT canonical_name, profile_text, training_date, version FROM profiles')
            rows = cursor.fetchall()
            
            if rows:
                for canonical_name, profile_text, training_date, version in rows:
                    self.user_profiles[canonical_name] = profile_text
                
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
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            
            training_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Сохраняем или обновляем профили
            for canonical_name, profile_text in self.user_profiles.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO profiles 
                    (canonical_name, profile_text, training_date, version, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (canonical_name, profile_text, training_date, PROFILES_VERSION))
            
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
            print(f"Портреты сохранены в базу данных {DB_FILE}")
        except Exception as e:
            print(f"Ошибка при сохранении портретов в базу данных: {e}")
    
    def _clear_profiles(self):
        """Очищает все профили из базы данных"""
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM profiles')
            cursor.execute('DELETE FROM metadata')
            conn.commit()
            conn.close()
            print("База данных очищена")
        except Exception as e:
            print(f"Ошибка при очистке базы данных: {e}")
    
    def create_all_profiles(self, force_recreate: bool = False):
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
                    return
        
        # Если портреты не загружены или требуется пересоздание
        if force_recreate or not profiles_loaded:
            if force_recreate:
                print("\nПринудительное пересоздание портретов...")
                # Очищаем старые профили
                self._clear_profiles()
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
    
    def find_user_by_name(self, name: str) -> Optional[str]:
        """Находит каноническое имя по частичному совпадению"""
        # Нормализуем имя
        canonical = self._normalize_name(name)
        if canonical and canonical in self.messages_by_user:
            return canonical
        
        # Если нормализация не помогла, ищем частичное совпадение
        name_lower = name.lower().strip()
        for canonical_name in self.messages_by_user.keys():
            if name_lower in canonical_name.lower() or canonical_name.lower() in name_lower:
                return canonical_name
            # Также проверяем варианты имен
            for variant in self.name_variants.get(canonical_name, []):
                if name_lower in variant.lower() or variant.lower() in name_lower:
                    return canonical_name
        return None
    
    def generate_response(self, question: str, canonical_name: str) -> str:
        """Генерирует ответ в стиле указанного участника"""
        profile = self.user_profiles.get(canonical_name, "")
        messages = self.messages_by_user.get(canonical_name, [])
        
        # Берем примеры сообщений для контекста
        examples = "\n".join([f"- {msg}" for msg in messages[-20:]])
        
        prompt = f"""Ты - {canonical_name}. Отвечай на вопрос в его стиле общения.

Портрет стиля {canonical_name}:
{profile}

Примеры сообщений {canonical_name}:
{examples}

Вопрос: {question}

Ответь на вопрос в стиле {canonical_name}, используя его манеру общения, типичные фразы и стиль. Ответ должен быть естественным и соответствовать его стилю."""
        
        response = self.call_ollama(prompt)
        return response
    
    def parse_question(self, user_input: str) -> tuple:
        """Парсит вопрос и извлекает имя участника из формата ^Имя^"""
        # Ищем паттерн ^Имя^
        pattern = r'\^([^^]+)\^'
        matches = re.findall(pattern, user_input)
        
        if matches:
            # Берем последнее совпадение
            name = matches[-1]
            # Удаляем ^Имя^ из вопроса
            question = re.sub(pattern, '', user_input).strip()
            return question, name
        else:
            return user_input, None
    
    def run(self, force_recreate: bool = False):
        """Запускает бота"""
        print("=" * 60)
        print("Telegram Bot - Генератор ответов в стиле участников")
        print("=" * 60)
        
        # Загружаем сообщения
        self.load_messages()
        
        # Создаем портреты (или загружаем сохраненные)
        self.create_all_profiles(force_recreate=force_recreate)
        
        print("\n" + "=" * 60)
        print("Бот готов! Задавайте вопросы в формате:")
        print("  Какая музыка тебе нравится ^Тимур^")
        print("  Что думаешь о погоде ^Юрий^")
        print("Для выхода введите 'exit' или 'quit'")
        print("=" * 60 + "\n")
        
        while True:
            try:
                user_input = input("Вопрос: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'выход']:
                    print("До свидания!")
                    break
                
                if not user_input:
                    continue
                
                # Парсим вопрос
                question, name = self.parse_question(user_input)
                
                if not name:
                    print("Ошибка: Укажите имя участника в формате ^Имя^")
                    print("Доступные участники:")
                    for canonical_name in self.messages_by_user.keys():
                        print(f"  - {canonical_name}")
                    continue
                
                # Находим участника
                canonical_name = self.find_user_by_name(name)
                
                if not canonical_name:
                    print(f"Ошибка: Участник '{name}' не найден")
                    print("Доступные участники:")
                    for available_name in self.messages_by_user.keys():
                        print(f"  - {available_name}")
                    continue
                
                print(f"\n[{canonical_name} отвечает...]")
                
                # Генерируем ответ
                response = self.generate_response(question, canonical_name)
                print(f"{canonical_name}: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nДо свидания!")
                break
            except Exception as e:
                print(f"Ошибка: {e}\n")

if __name__ == "__main__":
    # Проверяем аргументы командной строки для принудительного переобучения
    force_recreate = "--retrain" in sys.argv or "--force" in sys.argv or "-r" in sys.argv
    
    if force_recreate:
        print("⚠ Режим принудительного переобучения активирован")
        response = input("Вы уверены, что хотите переобучить модель? (yes/no): ").strip().lower()
        if response not in ['yes', 'y', 'да', 'д']:
            print("Отменено. Используются существующие портреты.")
            force_recreate = False
    
    bot = TelegramBot()
    bot.run(force_recreate=force_recreate)

