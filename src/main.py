#!/usr/bin/env python3
"""
Основная точка входа для Telegram бота
"""
import sys
import asyncio
from pathlib import Path

# Добавляем корневую директорию в путь для импортов
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from utils.logging_config import setup_logging
from core.profile_manager import ProfileManager
from embeddings.manager import EmbeddingManager
from database.connection import initialize_database
from tg_bot.bot import TelegramBot
from utils import get_logger

logger = get_logger(__name__)


def load_messages_from_json() -> dict:
    """
    Загружает сообщения из JSON файла

    Returns:
        Словарь с данными сообщений
    """
    import json
    from collections import defaultdict

    json_file = Path(settings.messages_file)
    if not json_file.exists():
        logger.error(f"Файл {settings.messages_file} не найден")
        return {}

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        messages_by_user = defaultdict(list)
        name_to_user_id = {}

        messages = data.get("messages", [])
        logger.info(f"Загрузка {len(messages)} сообщений из {settings.messages_file}")

        for msg in messages:
            # Пропускаем служебные сообщения
            if msg.get("type") != "message":
                continue

            from_name = msg.get("from")
            from_id = msg.get("from_id")

            if not from_name or not from_id:
                continue

            # Нормализуем имя (простая логика, будет улучшена)
            canonical_name = from_name.strip()

            # Обрабатываем текст сообщения
            text_raw = msg.get("text", "")
            if isinstance(text_raw, list):
                text_parts = []
                for item in text_raw:
                    if isinstance(item, dict) and item.get("type") == "plain":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                text = " ".join(text_parts).strip()
            else:
                text = str(text_raw).strip()

            # Извлекаем plain текст из text_entities
            text_entities = msg.get("text_entities", [])
            plain_parts = [entity.get("text", "") for entity in text_entities if entity.get("type") == "plain"]
            if plain_parts:
                text = " ".join(plain_parts).strip()

            # Пропускаем пустые сообщения
            if not text or len(text) < 2:
                continue

            messages_by_user[canonical_name].append(text)
            name_to_user_id[canonical_name] = int(str(from_id).lstrip('user'))

        logger.info(f"Обработано {len(messages_by_user)} участников")
        return {
            'messages_by_user': dict(messages_by_user),
            'name_to_user_id': name_to_user_id
        }

    except Exception as e:
        logger.error(f"Ошибка загрузки сообщений: {e}")
        return {}


def initialize_application():
    """Инициализация приложения"""
    # Настраиваем логирование
    setup_logging()

    # Инициализируем базу данных
    initialize_database()

    logger.info("✓ Приложение инициализировано")


def main():
    """Основная функция"""
    # Инициализация
    initialize_application()

    # Парсинг аргументов командной строки
    import argparse

    parser = argparse.ArgumentParser(description='Telegram Bot - Генератор ответов в стиле участников')
    parser.add_argument('--telegram', '-t', action='store_true', help='Запуск в режиме Telegram бота')
    parser.add_argument('--load-messages', '-l', action='store_true', help='Загрузить сообщения из JSON')
    parser.add_argument('--update-profiles', '-p', action='store_true', help='Обновить профили пользователей')
    parser.add_argument('--update-embeddings', '-e', action='store_true', help='Обновить эмбеддинги')
    parser.add_argument('--messages-file', '-f', type=str, help='Путь к файлу с сообщениями')
    parser.add_argument('--force', action='store_true', help='Принудительное обновление')

    args = parser.parse_args()

    # Обновляем настройки если указан файл
    if args.messages_file:
        settings.messages_file = args.messages_file

    try:
        # Загружаем данные если требуется
        data = {}
        if args.load_messages or args.telegram:
            data = load_messages_from_json()
            if not data:
                logger.error("Не удалось загрузить данные сообщений")
                if not args.telegram:
                    return

        # Инициализируем менеджеры
        profile_manager = ProfileManager()
        embedding_manager = EmbeddingManager()

        # Обновляем профили если требуется
        if args.update_profiles or args.force:
            logger.info("Обновление профилей пользователей...")
            profiles = profile_manager.create_all_profiles(
                messages_by_user=data.get('messages_by_user', {}),
                force_recreate=args.force,
                rebuild_embeddings=args.update_embeddings
            )
            logger.info(f"✓ Создано {len(profiles)} профилей")

        # Обновляем эмбеддинги если требуется
        if args.update_embeddings:
            logger.info("Обновление эмбеддингов...")
            if data.get('messages_by_user'):
                name_to_user_id = data.get('name_to_user_id', {})
                for canonical_name, messages in data['messages_by_user'].items():
                    user_id = name_to_user_id.get(canonical_name)
                    if user_id:
                        created = embedding_manager.create_embeddings_for_user(
                            user_id=user_id,
                            canonical_name=canonical_name,
                            messages=messages,
                            force_recreate=args.force
                        )
                        logger.info(f"✓ Создано {created} эмбеддингов для {canonical_name}")
            else:
                logger.warning("Нет данных для создания эмбеддингов")

        # Запуск Telegram бота
        if args.telegram:
            logger.info("Запуск Telegram бота...")
            bot = TelegramBot(
                profile_manager=profile_manager,
                embedding_manager=embedding_manager,
                messages_data=data
            )
            asyncio.run(bot.run())
        else:
            logger.info("Командная строка: выполнение завершено успешно")

    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания, завершение работы...")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()