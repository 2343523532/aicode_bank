import os

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession
from app.config import DATABASE_URL

# ---------------------------------------------------------------------------
# Database setup (async)
# ---------------------------------------------------------------------------
_current_database_url = DATABASE_URL


def _resolve_database_url() -> str:
    return os.environ.get("DATABASE_URL", DATABASE_URL)


async_engine = create_async_engine(_current_database_url, echo=False, future=True)
AsyncSessionLocal = sessionmaker(async_engine, expire_on_commit=False, class_=AsyncSession)


def refresh_engine_if_needed() -> None:
    global _current_database_url, async_engine, AsyncSessionLocal

    resolved_url = _resolve_database_url()
    if resolved_url == _current_database_url:
        return

    _current_database_url = resolved_url
    async_engine = create_async_engine(_current_database_url, echo=False, future=True)
    AsyncSessionLocal = sessionmaker(async_engine, expire_on_commit=False, class_=AsyncSession)


def get_async_engine():
    refresh_engine_if_needed()
    return async_engine

async def get_session() -> AsyncSession:
    refresh_engine_if_needed()
    async with AsyncSessionLocal() as session:
        yield session  # pragma: no cover - thin wrapper
