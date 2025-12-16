from __future__ import annotations

from sqlalchemy import select

from ai_business_assistant.shared.db.base import Base
from ai_business_assistant.shared.db.models import Role
from ai_business_assistant.shared.db.session import get_engine, get_sessionmaker


async def init_db() -> None:
    engine = get_engine()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = get_sessionmaker()
    async with session_maker() as session:
        existing = (await session.execute(select(Role.name))).scalars().all()
        for name in ("admin", "user"):
            if name not in existing:
                session.add(Role(name=name))
        await session.commit()
