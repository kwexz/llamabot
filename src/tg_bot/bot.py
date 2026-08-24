"""
Telegram бот с поддержкой асинхронной обработки
"""
import asyncio
import json
from typing import Dict, List, Optional

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from config.settings import settings
from core.profile_manager import ProfileManager
from embeddings.manager import EmbeddingManager
from database.repository import UserRepository, SettingsRepository
from utils import get_logger
from utils.security import SecurityManager

logger = get_logger(__name__)


class TelegramBot:
    """Telegram бот для генерации ответов в стиле участников"""

    def __init__(
        self,
        profile_manager: ProfileManager,
        embedding_manager: EmbeddingManager,
        messages_data: Dict
    ):
        """
        Args:
            profile_manager: Менеджер профилей
            embedding_manager: Менеджер эмбеддингов
            messages_data: Данные сообщений
        """
        self.profile_manager = profile_manager
        self.embedding_manager = embedding_manager
        self.messages_data = messages_data

        self.user_repo = UserRepository()
        self.settings_repo = SettingsRepository()
        self.security = SecurityManager()

        self.app: Optional[Application] = None
        self.bot_username: Optional[str] = None

        # Загружаем канонические имена
        self.canonical_names = self._load_canonical_names()
        self.name_to_user_id = self.messages_data.get('name_to_user_id', {})

    def _load_canonical_names(self) -> List[str]:
        """Загружает список канонических имен"""
        names_str = self.settings_repo.get_setting("canonical_names")
        if names_str:
            return [name.strip() for name in names_str.split(",") if name.strip()]
        return list(self.messages_data.get('messages_by_user', {}).keys())

    async def post_init(self, application: Application) -> None:
        """Пост-инициализация после создания приложения"""
        try:
            bot_info = await application.bot.get_me()
            self.bot_username = bot_info.username.lower() if bot_info.username else None
            logger.info(f"✓ Telegram бот инициализирован: @{bot_info.username}")
        except Exception as e:
            logger.error(f"Ошибка получения информации о боте: {e}")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик команды /start"""
        if not self._is_allowed_chat(update):
            return

        help_text = """👋 Привет! Я бот, который отвечает в стиле участников группы.

📝 Формат вопросов:
Какая музыка тебе нравится ^Алексей^
Что думаешь о погоде ^Мария^

Доступные участники:
"""
        for name in self.canonical_names[:10]:  # Показываем первых 10
            help_text += f"  • {name}\n"

        if len(self.canonical_names) > 10:
            help_text += f"  ... и ещё {len(self.canonical_names) - 10} участников\n"

        await update.message.reply_text(help_text)

    async def list_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик команды /list"""
        if not self._is_allowed_chat(update):
            return

        if not self.canonical_names:
            await update.message.reply_text("Участники ещё не загружены.")
            return

        list_text = "📋 Доступные участники:\n\n"
        for name in self.canonical_names:
            list_text += f"  • {name}\n"

        await update.message.reply_text(list_text)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик текстовых сообщений"""
        if not update.message or not update.message.text:
            return

        if not self._is_allowed_chat(update):
            return

        message_text = update.message.text.strip()
        user = update.message.from_user

        logger.info(f"[@{user.username or user.first_name}] {message_text}")

        try:
            # Парсим вопрос
            question, target_name = self._parse_question(message_text)

            if not target_name:
                # Проверяем, является ли это обращением к боту
                if self._is_bot_mentioned(update):
                    await self._show_help(update)
                return

            # Находим участника
            canonical_name = self._find_user_by_name(target_name)
            if not canonical_name:
                await self._show_user_not_found(update, target_name)
                return

            # Генерируем ответ
            await self._generate_and_send_response(update, question, canonical_name)

        except Exception as e:
            logger.error(f"Ошибка обработки сообщения: {e}")
            await update.message.reply_text("Произошла ошибка при обработке запроса.")

    def _parse_question(self, message_text: str) -> tuple[str, Optional[str]]:
        """Парсит вопрос и извлекает имя участника"""
        import re

        # Ищем паттерн ^Имя^
        pattern = r'\^([^^]+)\^'
        matches = re.findall(pattern, message_text)

        if matches:
            # Берем последнее совпадение
            name = matches[-1]
            # Удаляем ^Имя^ из вопроса
            question = re.sub(pattern, '', message_text).strip()
            return question, name

        return message_text, None

    def _find_user_by_name(self, name: str) -> Optional[str]:
        """Находит каноническое имя по имени/алиасу"""
        # Проверяем прямое совпадение
        if name in self.canonical_names:
            return name

        # Ищем по алиасам
        return self.user_repo.find_user_by_alias(name.lower())

    def _is_allowed_chat(self, update: Update) -> bool:
        """Проверяет, разрешен ли чат"""
        chat_id = update.effective_chat.id

        if not settings.allowed_chat_ids:
            return True  # Все чаты разрешены

        return chat_id in settings.allowed_chat_ids

    def _is_bot_mentioned(self, update: Update) -> bool:
        """Проверяет, упомянут ли бот"""
        if not self.bot_username:
            return False

        if update.effective_chat.type == "private":
            return True

        if update.message.entities:
            for entity in update.message.entities:
                if entity.type == "mention":
                    mention_text = update.message.text[entity.offset:entity.offset + entity.length]
                    if mention_text.lower().lstrip('@') == self.bot_username:
                        return True

        return False

    async def _generate_and_send_response(self, update: Update, question: str, canonical_name: str) -> None:
        """Генерирует и отправляет ответ"""
        # Отправляем сообщение о обработке
        processing_msg = await update.message.reply_text(f"🤔 [{canonical_name} думает...]")

        try:
            # Получаем профиль
            profile = self.profile_manager.get_profile(canonical_name)
            if not profile:
                await processing_msg.edit_text(f"❌ Профиль для {canonical_name} не найден.")
                return

            # Ищем релевантные сообщения
            user_id = self.name_to_user_id.get(canonical_name)
            relevant_messages = []
            if user_id:
                relevant_messages = self.embedding_manager.retrieve_relevant_messages(user_id, question)

            # Генерируем ответ
            response = await self.profile_manager.generate_response(
                question=question,
                canonical_name=canonical_name,
                profile=profile.profile_text,
                relevant_messages=relevant_messages
            )

            # Отправляем ответ
            formatted_response = f"💬 <b>{canonical_name}:</b>\n\n{response}"
            await processing_msg.edit_text(formatted_response, parse_mode='HTML')

            logger.info(f"✓ Ответ от {canonical_name} отправлен")

        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            await processing_msg.edit_text("❌ Произошла ошибка при генерации ответа.")

    async def _show_help(self, update: Update) -> None:
        """Показывает справку"""
        help_text = """❓ Для получения ответа укажите имя участника в формате:

<b>Вопрос ^Имя^</b>

Например:
<code>Какая музыка тебе нравится ^Алексей^</code>

Используйте команду /list для просмотра доступных участников."""
        await update.message.reply_text(help_text, parse_mode='HTML')

    async def _show_user_not_found(self, update: Update, name: str) -> None:
        """Показывает сообщение о не найденном пользователе"""
        error_text = f"❌ Участник '{name}' не найден.\n\nДоступные участники:\n"
        for canonical_name in self.canonical_names[:10]:
            error_text += f"  • {canonical_name}\n"

        if len(self.canonical_names) > 10:
            error_text += f"  ... и ещё {len(self.canonical_names) - 10} участников\n"

        error_text += "\nИспользуйте команду /list для полного списка."
        await update.message.reply_text(error_text)

    def setup_handlers(self) -> None:
        """Настраивает обработчики команд"""
        if not self.app:
            raise RuntimeError("Приложение не инициализировано")

        # Команды
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help", self.start_command))
        self.app.add_handler(CommandHandler("list", self.list_command))

        # Текстовые сообщения
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        # Пост-инициализация
        self.app.post_init = self.post_init

    async def run(self) -> None:
        """Запускает бота"""
        if not settings.telegram_bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN не задан в настройках")

        self.app = Application.builder().token(settings.telegram_bot_token).build()
        self.setup_handlers()

        logger.info("🚀 Запуск Telegram бота...")

        try:
            await self.app.initialize()
            await self.app.start()

            # Запускаем polling
            await self.app.updater.start_polling()
            logger.info("✓ Telegram бот запущен и слушает сообщения")

            # Держим приложение запущенным
            await asyncio.Event().wait()

        except KeyboardInterrupt:
            logger.info("Получен сигнал прерывания...")
        except Exception as e:
            logger.error(f"Ошибка запуска бота: {e}")
            raise
        finally:
            if self.app:
                await self.app.stop()
                await self.app.shutdown()
                logger.info("✓ Telegram бот остановлен")