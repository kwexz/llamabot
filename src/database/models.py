"""
Модели данных для работы с базой данных
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel


class UserProfile(BaseModel):
    """Модель профиля пользователя"""
    user_id: int
    canonical_name: str
    profile_text: str
    training_date: str
    version: str
    updated_at: Optional[datetime] = None


class Alias(BaseModel):
    """Модель алиаса пользователя"""
    id: Optional[int] = None
    user_id: int
    alias: str
    created_at: Optional[datetime] = None


class MessageEmbedding(BaseModel):
    """Модель эмбеддинга сообщения"""
    id: Optional[str] = None
    canonical_name: str
    message_text: str
    embedding: Optional[List[float]] = None
    timestamp: Optional[datetime] = None
    source: str = "manual"


class Settings(BaseModel):
    """Модель настроек"""
    key: str
    value: str


class Metadata(BaseModel):
    """Модель метаданных"""
    key: str
    value: str


class TelegramMessage(BaseModel):
    """Модель сообщения Telegram"""
    type: str = "message"
    from_name: Optional[str] = None
    from_id: Optional[str] = None
    text: Optional[str] = None
    text_entities: Optional[List[Dict[str, Any]]] = None
    date: Optional[str] = None


class ChatData(BaseModel):
    """Модель данных чата"""
    name: str
    type: str
    id: int
    messages: List[TelegramMessage]


class JsonExportData(BaseModel):
    """Модель данных экспорта JSON"""
    messages: List[Dict[str, Any]]