from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any

from sqlalchemy import DateTime, String, Text, func, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.config.config import DatabaseSettings, get_settings


class Base(DeclarativeBase):
    pass


class ConversationMessageRow(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(24))
    content: Mapped[str] = mapped_column(Text())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class PostgresClient:
    def __init__(
        self,
        *,
        settings: DatabaseSettings,
        echo: bool = False,
        pool_size: int = 10,
        max_overflow: int = 20,
    ) -> None:
        self._settings = settings
        self._echo = echo
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._engine: AsyncEngine | None = None
        self._session_maker: async_sessionmaker[AsyncSession] | None = None

    @property
    def is_connected(self) -> bool:
        return self._engine is not None and self._session_maker is not None

    async def connect(self) -> None:
        if self.is_connected:
            return

        self._engine = create_async_engine(
            self._settings.sqlalchemy_url,
            echo=self._echo,
            pool_pre_ping=True,
            pool_size=self._pool_size,
            max_overflow=self._max_overflow,
        )
        self._session_maker = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
        await self.ensure_schema()

    async def close(self) -> None:
        if self._engine is None:
            return
        await self._engine.dispose()
        self._engine = None
        self._session_maker = None

    async def ensure_schema(self) -> None:
        engine = self._require_engine()
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def session(self) -> AsyncIterator[AsyncSession]:
        session_maker = self._require_session_maker()
        async with session_maker() as session:
            yield session

    def _require_engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("SQLAlchemy engine is not initialized. Call connect() first.")
        return self._engine

    def _require_session_maker(self) -> async_sessionmaker[AsyncSession]:
        if self._session_maker is None:
            raise RuntimeError(
                "SQLAlchemy session maker is not initialized. Call connect() first."
            )
        return self._session_maker


@dataclass(kw_only=True)
class ConversationService:
    db: PostgresClient

    async def save_message(self, *, conversation_id: str, role: str, content: str) -> None:
        async for session in self.db.session():
            session.add(
                ConversationMessageRow(
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                )
            )
            await session.commit()

    async def get_history(
        self,
        *,
        conversation_id: str,
        limit: int = 50,
        before: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Most recent messages, oldest first.
         pages further back in time: pass the created_at of the
        oldest message seen so far to get the page right before it.
        """
        stmt = select(
            ConversationMessageRow.role,
            ConversationMessageRow.content,
            ConversationMessageRow.created_at,
        ).where(ConversationMessageRow.conversation_id == conversation_id)
        if before is not None:
            stmt = stmt.where(ConversationMessageRow.created_at < before)
        stmt = stmt.order_by(ConversationMessageRow.created_at.desc()).limit(limit)

        async for session in self.db.session():
            result = await session.execute(stmt)
            rows = result.all()
            history = [
                {"role": row.role, "content": row.content, "created_at": row.created_at}
                for row in rows
            ]
            return list(reversed(history))

        return []


@lru_cache
def get_postgres_client() -> PostgresClient:
    settings = get_settings()
    return PostgresClient(settings=settings.db_settings)


def get_service(*, database: PostgresClient) -> ConversationService:
    return ConversationService(db=database)
