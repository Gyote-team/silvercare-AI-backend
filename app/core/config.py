"""환경 변수와 .env 파일에서 서버 설정을 읽는다."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """환경 변수와 .env 파일에서 읽는 서버 설정."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str
    ai_internal_token: SecretStr = Field(min_length=1)
    storage_local_root: str = Field(min_length=1)
    document_render_dpi: int = Field(default=150, gt=0)
    document_max_pages: int = Field(default=30, gt=0)
    document_max_page_side_px: int = Field(default=2048, gt=0)
    openai_api_key: SecretStr | None = None
    openai_vlm_model: str = Field(default="gpt-5.6-sol", min_length=1)
    openai_vlm_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "high"
    openai_vlm_max_output_tokens: int = Field(default=16000, gt=0)
    openai_vlm_timeout_seconds: float = Field(default=180, gt=0)
    openai_vlm_connect_timeout_seconds: float = Field(default=10, gt=0)
    openai_vlm_max_retries: int = Field(default=2, ge=0)
    openai_vlm_retry_backoff_seconds: float = Field(default=1.0, ge=0)


@lru_cache
def get_settings() -> Settings:
    """설정을 한 번만 읽어 재사용한다."""
    return Settings()
