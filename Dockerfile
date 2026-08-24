# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем системные зависимости для ChromaDB и других компонентов
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создаем пользователя для безопасности
RUN useradd --create-home --shell /bin/bash app

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY src/ ./src/
COPY .env.example ./.env

# Создаем директории для данных
RUN mkdir -p logs chroma_db data && \
    chown -R app:app /app

# Переключаемся на обычного пользователя
USER app

# Устанавливаем переменные окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Добавляем healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Запуск приложения
CMD ["python", "src/main.py", "--telegram"]