from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai_business_assistant.auth_service.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from ai_business_assistant.shared.config import get_settings
from ai_business_assistant.shared.db.models import Role, User
from ai_business_assistant.shared.db.session import get_db_session
from ai_business_assistant.shared.security.dependencies import get_current_token
from ai_business_assistant.shared.security.jwt import create_access_token
from ai_business_assistant.shared.security.passwords import hash_password, verify_password


router = APIRouter(prefix="/auth", tags=["auth"])


async def _user_to_response(user: User) -> UserResponse:
    roles = [r.name for r in user.roles]
    return UserResponse(id=user.id, email=user.email, roles=roles)


@router.post("/register", response_model=UserResponse)
async def register(payload: RegisterRequest, db: AsyncSession = Depends(get_db_session)) -> UserResponse:
    existing = (await db.execute(select(User).where(User.email == payload.email))).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    role_user = (await db.execute(select(Role).where(Role.name == "user"))).scalar_one_or_none()
    if role_user is None:
        role_user = Role(name="user")
        db.add(role_user)
        await db.flush()

    user = User(email=payload.email, password_hash=hash_password(payload.password), roles=[role_user])
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return await _user_to_response(user)


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db_session)) -> TokenResponse:
    user = (await db.execute(select(User).where(User.email == payload.email))).scalar_one_or_none()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    settings = get_settings()
    token = create_access_token(
        user_id=str(user.id),
        email=user.email,
        roles=[r.name for r in user.roles],
        secret_key=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
        expires_in_seconds=settings.jwt_access_token_ttl_seconds,
    )
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserResponse)
async def me(token=Depends(get_current_token), db: AsyncSession = Depends(get_db_session)) -> UserResponse:
    user_id = uuid.UUID(token.user_id)
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return await _user_to_response(user)
