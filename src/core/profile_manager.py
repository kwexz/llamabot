"""
Менеджер профилей пользователей с анализом сообщений
"""
from typing import List, Dict, Optional
from datetime import datetime
import json

from config.settings import settings
from database.repository import UserRepository, SettingsRepository, MetadataRepository
from ai.ollama_client import OllamaClient
from utils import TextAnalyzer, get_logger
from models import UserProfile

logger = get_logger(__name__)


class ProfileManager:
    """Менеджер для создания и управления профилями пользователей"""

    def __init__(self):
        self.user_repo = UserRepository()
        self.settings_repo = SettingsRepository()
        self.metadata_repo = MetadataRepository()
        self.ollama_client = OllamaClient()

    def create_profile(self, canonical_name: str, messages: List[str]) -> str:
        """
        Создает портрет участника на основе его сообщений

        Args:
            canonical_name: Каноническое имя пользователя
            messages: Список сообщений пользователя

        Returns:
            Текст профиля пользователя
        """
        if not messages:
            logger.warning(f"Нет сообщений для создания профиля {canonical_name}")
            return ""

        # Фильтруем технические сообщения
        technical_phrases = {
            "ок", "окей", "спасибо", "спс", "thx", "ок.", "ок!", "спасибо!", "спасибо."
        }

        def is_technical(msg: str) -> bool:
            text = msg.strip().lower()
            if not text or text in technical_phrases:
                return True
            if len(text) <= 2:
                return True
            return False

        filtered = [m for m in messages if not is_technical(m)]
        if not filtered:
            filtered = messages

        # Убираем дубликаты и сортируем по длине
        seen = set()
        unique_messages = []
        for m in filtered:
            if m not in seen:
                seen.add(m)
                unique_messages.append(m)

        unique_messages.sort(key=lambda x: len(x), reverse=True)
        selected_messages = unique_messages[:50]  # Ограничиваем для анализа

        # Анализируем частотные слова
        analyzer = TextAnalyzer(selected_messages)
        common_words = analyzer.most_common_words(top_n=30)
        common_words_str = "\n".join(
            [f"- {word} ({count})" for word, count in common_words]
        ) or "- (недостаточно данных)"

        # Формируем примеры сообщений
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

        profile = self.ollama_client.generate_text(prompt)
        logger.info(f"✓ Создан профиль для {canonical_name} ({len(profile)} символов)")
        return profile

    def save_profile(self, profile: UserProfile) -> bool:
        """
        Сохраняет профиль пользователя в базу данных

        Args:
            profile: Профиль пользователя

        Returns:
            True при успехе
        """
        return self.user_repo.save_profile(profile)

    def get_profile(self, canonical_name: str) -> Optional[UserProfile]:
        """
        Получает профиль пользователя из базы данных

        Args:
            canonical_name: Каноническое имя пользователя

        Returns:
            Профиль пользователя или None
        """
        return self.user_repo.get_profile(canonical_name)

    def load_profiles_from_db(self) -> Dict[str, str]:
        """
        Загружает все профили из базы данных

        Returns:
            Словарь canonical_name -> profile_text
        """
        profiles = {}
        for profile in self.user_repo.get_all_profiles():
            profiles[profile.canonical_name] = profile.profile_text
        return profiles

    def create_all_profiles(
        self,
        messages_by_user: Dict[str, List[str]],
        force_recreate: bool = False,
        rebuild_embeddings: bool = False
    ) -> Dict[str, str]:
        """
        Создает профили для всех пользователей

        Args:
            messages_by_user: Словарь canonical_name -> messages
            force_recreate: Принудительно пересоздать все профили
            rebuild_embeddings: Пересоздать эмбеддинги

        Returns:
            Словарь canonical_name -> profile_text
        """
        profiles = {}

        # Пытаемся загрузить существующие профили
        if not force_recreate:
            existing_profiles = self.load_profiles_from_db()
            if existing_profiles:
                # Проверяем, все ли пользователи имеют профили
                missing_profiles = [
                    name for name in messages_by_user.keys()
                    if name not in existing_profiles
                ]

                if missing_profiles:
                    logger.info(f"Найдены новые участники без профилей: {', '.join(missing_profiles)}")
                    # Создаем профили только для новых
                    for canonical_name in missing_profiles:
                        logger.info(f"Анализ стиля {canonical_name}...")
                        profile_text = self.create_profile(canonical_name, messages_by_user[canonical_name])
                        profiles[canonical_name] = profile_text

                        # Сохраняем профиль
                        user_id = self.user_repo.get_name_to_user_id_mapping().get(canonical_name)
                        if user_id:
                            profile = UserProfile(
                                user_id=user_id,
                                canonical_name=canonical_name,
                                profile_text=profile_text,
                                training_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                version=settings.profiles_version
                            )
                            self.save_profile(profile)
                            logger.info(f"✓ {canonical_name}: {profile_text[:100]}...")

                    # Обновляем метаданные
                    self.metadata_repo.save_metadata("training_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    return {**existing_profiles, **profiles}

                logger.info("✓ Используются сохраненные профили участников")
                return existing_profiles

        # Создаем профили заново
        if force_recreate:
            logger.info("Принудительное пересоздание портретов...")
            self.user_repo.clear_profiles()

        logger.info("Создание портретов участников (первый запуск)...")

        for canonical_name, messages in messages_by_user.items():
            logger.info(f"Анализ стиля {canonical_name}...")
            profile_text = self.create_profile(canonical_name, messages)
            profiles[canonical_name] = profile_text

            # Сохраняем профиль
            user_id = self.user_repo.get_name_to_user_id_mapping().get(canonical_name)
            if user_id:
                profile = UserProfile(
                    user_id=user_id,
                    canonical_name=canonical_name,
                    profile_text=profile_text,
                    training_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    version=settings.profiles_version
                )
                self.save_profile(profile)
                logger.info(f"✓ {canonical_name}: {profile_text[:100]}...")

        # Сохраняем метаданные
        self.metadata_repo.save_metadata("training_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.metadata_repo.save_metadata("version", settings.profiles_version)

        return profiles

    async def generate_response(
        self,
        question: str,
        canonical_name: str,
        profile: str,
        relevant_messages: Optional[List[tuple]] = None
    ) -> str:
        """
        Генерирует ответ в стиле указанного участника

        Args:
            question: Вопрос пользователя
            canonical_name: Каноническое имя пользователя
            profile: Профиль пользователя
            relevant_messages: Релевантные сообщения для RAG

        Returns:
            Сгенерированный ответ
        """
        # Формируем контекст из релевантных сообщений
        if relevant_messages:
            context_messages = "\n".join([f"- {msg}" for msg, _ in relevant_messages])
            context_info = f"Релевантные сообщения {canonical_name} по теме:\n{context_messages}"
        else:
            context_info = f"Общие сообщения {canonical_name}"

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
   * **Словарный запас и жаргон:** Специфические слова, ругательства, профессиональные термины.
   * **Речевые паттерны:** Длина предложений (короткие/длинные), риторические приемы, повторы.
   * **Интонация и эмоции:** Самоуверенность, ирония, добродушие, грусть и т.д.
   * **Уникальные выражения-фишки:** Известные фразы или "пунктики" персонажа.
3. **Структура:** Ответ должен быть целостным, не нужно помечать его как "Ответ:" или добавлять мета-комментарии."""

        response = await self.ollama_client.generate_text(prompt)
        logger.debug(f"Сгенерирован ответ для {canonical_name}: {response[:100]}...")
        return response