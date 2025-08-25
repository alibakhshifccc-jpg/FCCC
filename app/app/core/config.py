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


class setting:
    def __init__(self)->Noen:
        pass