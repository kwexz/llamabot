"""
Модуль безопасности для валидации входных данных и защиты от атак
"""
import re
import html
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from . import get_logger

logger = get_logger(__name__)


class InputValidator:
    """Валидатор входных данных"""

    # Регулярные выражения для валидации
    USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_]{5,32}$')
    TELEGRAM_ID_PATTERN = re.compile(r'^-?\d{5,15}$')
    CANONICAL_NAME_PATTERN = re.compile(r'^[a-zA-Zа-яА-ЯёЁ\s\-]{2,50}$')

    # Ограничения длин
    MAX_MESSAGE_LENGTH = 4000
    MAX_PROMPT_LENGTH = 8000
    MAX_USERNAME_LENGTH = 32
    MAX_CANONICAL_NAME_LENGTH = 50

    # Список запрещенных слов и паттернов
    FORBIDDEN_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # XSS попытки
        r'javascript:',                # JavaScript URL
        r'data:',                      # Data URL
        r'vbscript:',                  # VBScript
        r'on\w+\s*=',                  # Event handlers
    ]

    @staticmethod
    def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
        """
        Очищает текст от потенциально опасного содержимого

        Args:
            text: Исходный текст
            max_length: Максимальная длина (если None, используется MAX_MESSAGE_LENGTH)

        Returns:
            Очищенный текст
        """
        if not text:
            return ""

        # Экранируем HTML
        cleaned = html.escape(text)

        # Удаляем потенциально опасные паттерны
        for pattern in InputValidator.FORBIDDEN_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)

        # Ограничиваем длину
        max_len = max_length or InputValidator.MAX_MESSAGE_LENGTH
        if len(cleaned) > max_len:
            cleaned = cleaned[:max_len] + "..."
            logger.warning(f"Текст обрезан до {max_len} символов")

        return cleaned

    @staticmethod
    def validate_username(username: str) -> bool:
        """
        Валидирует имя пользователя Telegram

        Args:
            username: Имя пользователя

        Returns:
            True если валидно
        """
        if not username or len(username) > InputValidator.MAX_USERNAME_LENGTH:
            return False

        return bool(InputValidator.USERNAME_PATTERN.match(username))

    @staticmethod
    def validate_telegram_id(telegram_id: int) -> bool:
        """
        Валидирует Telegram ID

        Args:
            telegram_id: ID пользователя или чата

        Returns:
            True если валидно
        """
        return bool(InputValidator.TELEGRAM_ID_PATTERN.match(str(telegram_id)))

    @staticmethod
    def validate_canonical_name(name: str) -> bool:
        """
        Валидирует каноническое имя пользователя

        Args:
            name: Каноническое имя

        Returns:
            True если валидно
        """
        if not name or len(name) > InputValidator.MAX_CANONICAL_NAME_LENGTH:
            return False

        return bool(InputValidator.CANONICAL_NAME_PATTERN.match(name))

    @staticmethod
    def validate_prompt_length(prompt: str) -> bool:
        """
        Проверяет длину промпта

        Args:
            prompt: Промпт для AI

        Returns:
            True если длина допустимая
        """
        return len(prompt) <= InputValidator.MAX_PROMPT_LENGTH

    @staticmethod
    def is_safe_url(url: str) -> bool:
        """
        Проверяет, является ли URL безопасным

        Args:
            url: URL для проверки

        Returns:
            True если безопасный
        """
        try:
            parsed = urlparse(url)
            # Разрешаем только HTTP/HTTPS
            if parsed.scheme not in ['http', 'https']:
                return False
            # Проверяем на наличие потенциально опасных символов
            if any(char in url for char in ['<', '>', '"', "'"]):
                return False
            return True
        except Exception:
            return False


class RateLimiter:
    """Ограничитель частоты запросов"""

    def __init__(self, max_requests: int = 10, time_window: int = 60):
        """
        Args:
            max_requests: Максимальное количество запросов
            time_window: Временное окно в секундах
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, List[float]] = {}

    def is_allowed(self, key: str) -> bool:
        """
        Проверяет, разрешен ли запрос

        Args:
            key: Ключ для идентификации пользователя/запроса

        Returns:
            True если запрос разрешен
        """
        import time
        current_time = time.time()

        if key not in self.requests:
            self.requests[key] = []

        # Очищаем старые запросы
        self.requests[key] = [
            timestamp for timestamp in self.requests[key]
            if current_time - timestamp < self.time_window
        ]

        # Проверяем лимит
        if len(self.requests[key]) >= self.max_requests:
            logger.warning(f"Превышен лимит запросов для ключа {key}")
            return False

        # Добавляем текущий запрос
        self.requests[key].append(current_time)
        return True

    def get_remaining_requests(self, key: str) -> int:
        """
        Получает количество оставшихся запросов

        Args:
            key: Ключ пользователя

        Returns:
            Количество оставшихся запросов
        """
        if key not in self.requests:
            return self.max_requests

        import time
        current_time = time.time()

        # Очищаем старые запросы
        self.requests[key] = [
            timestamp for timestamp in self.requests[key]
            if current_time - timestamp < self.time_window
        ]

        return max(0, self.max_requests - len(self.requests[key]))

    def reset(self, key: str) -> None:
        """
        Сбрасывает счетчик для ключа

        Args:
            key: Ключ для сброса
        """
        if key in self.requests:
            del self.requests[key]


class SecurityManager:
    """Центральный менеджер безопасности"""

    def __init__(self):
        self.validator = InputValidator()
        self.rate_limiter = RateLimiter()

    def validate_and_sanitize_message(self, message: str, user_key: str) -> tuple[bool, str]:
        """
        Валидирует и очищает сообщение

        Args:
            message: Исходное сообщение
            user_key: Ключ пользователя для rate limiting

        Returns:
            (разрешено, очищенное_сообщение)
        """
        # Проверяем rate limit
        if not self.rate_limiter.is_allowed(user_key):
            return False, "Превышен лимит запросов. Попробуйте позже."

        # Очищаем и валидируем текст
        sanitized = self.validator.sanitize_text(message)

        if not sanitized:
            return False, "Сообщение пустое или содержит недопустимый контент."

        return True, sanitized

    def validate_telegram_user(self, user_data: Dict[str, Any]) -> bool:
        """
        Валидирует данные пользователя Telegram

        Args:
            user_data: Данные пользователя

        Returns:
            True если данные валидны
        """
        user_id = user_data.get('id')
        username = user_data.get('username')

        if not self.validator.validate_telegram_id(user_id):
            logger.warning(f"Невалидный Telegram ID: {user_id}")
            return False

        if username and not self.validator.validate_username(username):
            logger.warning(f"Невалидное имя пользователя: {username}")
            return False

        return True