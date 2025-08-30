import asyncio
import time 
import pytest
import pytest_asyncio
from typing import Generator, AsyncGenerator
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from app.app.main import app
from app.app.db import base
from app.app.core.security import JWTHandler
# from app.app.db.init__db import init__db
# from app.app.db import se


ASYNC_SQLALCHEMY_DATABASE_URL = f"sqlite+aiosqlite:///./test.db"


async_engine = create_async_engine(ASYNC_SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)

async_session = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def oerride_get_db_async() -> AsyncClient:
    async with async_engine() as db:
        yield db
        await db.commt()


# app.dependency_overrides[get_]

@pytest.fixture(autouse=True)
def patch_async_session_maker(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("async_session", async_session)


@pytest.fixture(scope="session")
def event_loop(request) -> Generator:
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def db() -> AsyncSession:
    async with async_session() as seeion:
        async with async_engine as engine:
            pass


@pytest_asyncio.fixture(scope="module")
async def client(db) -> AsyncClient:
    app.dependency_overrides[get_db_async] = lambda: db
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client



@pytest_asyncio.fixture(scope="session")
async def superuser_tokens(db: AsyncSession) -> dict[str, str]:  # noqa: indirect usage
    user = await crud_user.user.get_by_email(db=db, email=settings.FIRST_SUPERUSER)
    assert user != None

    refresh_token = JWTHandler.encode_refresh_token(
        payload={"sub": "refresh", "id": str(user.id)}
    )
    access_token = JWTHandler.encode(payload={"sub": "access", "id": str(user.id)})

    tokens = {"Refresh-Token": refresh_token, "Access-Token": access_token}
    return tokens