from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai_business_assistant.assistant_service.schemas import (
    ConversationDetailResponse,
    ConversationResponse,
    CreateConversationRequest,
    CreateMessageRequest,
    MessageResponse,
)
from ai_business_assistant.shared.db.models import Conversation, Message
from ai_business_assistant.shared.db.session import get_db_session
from ai_business_assistant.shared.security.dependencies import get_current_token


router = APIRouter(prefix="/assistant", tags=["assistant"])


def _message_to_response(message: Message) -> MessageResponse:
    return MessageResponse(
        id=message.id,
        conversation_id=message.conversation_id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
    )


def _conversation_to_response(conv: Conversation) -> ConversationResponse:
    return ConversationResponse(id=conv.id, title=conv.title, created_at=conv.created_at)


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    payload: CreateConversationRequest,
    token=Depends(get_current_token),
    db: AsyncSession = Depends(get_db_session),
) -> ConversationResponse:
    conv = Conversation(user_id=uuid.UUID(token.user_id), title=payload.title)
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return _conversation_to_response(conv)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: uuid.UUID,
    token=Depends(get_current_token),
    db: AsyncSession = Depends(get_db_session),
) -> ConversationDetailResponse:
    conv = (
        await db.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .where(Conversation.user_id == uuid.UUID(token.user_id))
        )
    ).scalar_one_or_none()

    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationDetailResponse(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        messages=[_message_to_response(m) for m in conv.messages],
    )


@router.post("/conversations/{conversation_id}/messages", response_model=MessageResponse)
async def add_message(
    conversation_id: uuid.UUID,
    payload: CreateMessageRequest,
    token=Depends(get_current_token),
    db: AsyncSession = Depends(get_db_session),
) -> MessageResponse:
    conv = (
        await db.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .where(Conversation.user_id == uuid.UUID(token.user_id))
        )
    ).scalar_one_or_none()

    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msg = Message(conversation_id=conv.id, role=payload.role, content=payload.content)
    db.add(msg)
    await db.commit()
    await db.refresh(msg)

    try:
        from ai_business_assistant.worker.celery_app import celery_app

        celery_app.send_task(
            "ai_business_assistant.worker.tasks.embed_message",
            kwargs={"message_id": str(msg.id), "content": msg.content},
        )
    except Exception:
        # In local/dev the broker may be unavailable; the API still succeeds.
        pass

    return _message_to_response(msg)
