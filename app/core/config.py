"""환경 변수와 .env 파일에서 서버 설정을 읽는다."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """환경 변수와 .env 파일에서 읽는 서버 설정."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str
    ai_internal_token: SecretStr = Field(min_length=1)
    storage_local_root: str = Field(min_length=1)


@lru_cache
def get_settings() -> Settings:
    """설정을 한 번만 읽어 재사용한다."""
    return Settings()
