from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class CreateConversationRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ConversationResponse(BaseModel):
    id: uuid.UUID
    title: str | None
    created_at: datetime


class CreateMessageRequest(BaseModel):
    role: str = Field(default="user", pattern="^(user|assistant|system)$")
    content: str = Field(min_length=1)


class MessageResponse(BaseModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    role: str
    content: str
    created_at: datetime


class ConversationDetailResponse(ConversationResponse):
    messages: list[MessageResponse]
