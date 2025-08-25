from typing import Any
from pydantic import (
    AnyHttpUrl,
    EmailStr,
    PostgresDsn,
    RedisDsn,
    field_validator
)
from pydantic_settings import BaseSettings, SettingsConfigDict


REFRESH_TOKEN_BLOCKLIST_KEY = "refresh_token_blocklist:{token}"
ACCESS_TOKEN_BLOCKLIST_KEY = "access_token_blocklist:{token}"




class AsyncPostgresDsn(PostgresDsn):
    allowed_schemes = {"postgres+asyncpg", "postgresql+asyncpg"}


class setting(BaseSettings):
    def __init__(self)-> None:
        pass

    PROJECT_NAME: str
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    SECRET_KEY: str
    BACKEND_CORS_ORIGINS: list[AnyHttpUrl] | str = []
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    REFRESH_TOKEN_EXPIRE_MINUTES: int
    JWT_ALGORITHM: str = "HS256" 