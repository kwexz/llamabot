class TextAnalyzer:
    """Простой анализатор текста для вычисления частот слов."""

    def __init__(self, messages):
        """
        Инициализация анализатора.

        Args:
            messages (list): Список сообщений для анализа.
        """
        self.messages = messages

    def most_common_words(self, top_n=10):
        """
        Получить наиболее частотные слова.

        Args:
            top_n (int): Количество наиболее частотных слов для возврата.

        Returns:
            list: Список кортежей (слово, частота).
        """
        from collections import Counter
        import re

        # Объединить все сообщения в один текст
        all_text = ' '.join(msg.get('text', '') for msg in self.messages if msg.get('text'))

        # Разделить на слова, убрать пунктуацию и привести к нижнему регистру
        words = re.findall(r'\b\w+\b', all_text.lower())

        # Подсчитать частоту слов
        word_counts = Counter(words)

        # Вернуть наиболее частотные слова
        return word_counts.most_common(top_n)