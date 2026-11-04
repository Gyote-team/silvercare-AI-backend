"""환경 변수와 .env 파일에서 서버 설정을 읽는다."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """환경 변수와 .env 파일에서 읽는 서버 설정."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str
    ai_internal_token: SecretStr = Field(min_length=1)
    storage_local_root: str = Field(min_length=1)
    llm_provider: str = "openai"
    # 결과 문서 작업이 아닌 VLM 파이프라인도 이 설정을 사용하므로, 실제 호출 시점에만 키를 검증한다.
    llm_api_key: SecretStr | None = None
    llm_base_url: str = Field(
        default="https://api.openai.com/v1", min_length=1
    )
    llm_model: str = Field(default="gpt-5.6-sol", min_length=1)
    llm_timeout_seconds: float = Field(default=90, gt=0, le=120)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_temperature: float = Field(default=0.2, ge=0, le=1)
    llm_reasoning_effort: str = "low"
    document_render_dpi: int = Field(default=150, gt=0)
    document_max_pages: int = Field(default=30, gt=0)
    document_max_page_side_px: int = Field(default=2048, gt=0)
    document_analysis_stale_minutes: int = Field(default=360, gt=0)
    document_analysis_max_workers: int = Field(default=2, gt=0)
    document_analysis_resume_on_startup: bool = True
    document_analysis_resume_pending_minutes: int = Field(default=10, gt=0)
    document_analysis_resume_limit: int = Field(default=100, gt=0)
    openai_api_key: SecretStr | None = None
    stt_api_key: SecretStr | None = None
    stt_base_url: str = "https://api.openai.com/v1"
    stt_model: str = "gpt-transcribe"
    stt_timeout_seconds: float = Field(default=180, gt=0, le=300)
    stt_lease_seconds: int = Field(default=600, gt=0)
    stt_db_retry_seconds: float = Field(default=5, gt=0, le=60)
    stt_max_transcript_chars: int = Field(default=100000, gt=0, le=1000000)
    openai_vlm_model: str = Field(default="gpt-5.6-sol", min_length=1)
    openai_vlm_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "high"
    openai_vlm_max_output_tokens: int = Field(default=16000, gt=0)
    openai_vlm_timeout_seconds: float = Field(default=180, gt=0)
    openai_vlm_connect_timeout_seconds: float = Field(default=10, gt=0)
    openai_vlm_max_retries: int = Field(default=2, ge=0)
    openai_vlm_retry_backoff_seconds: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def validate_stt_lease(self):
        """전사 전체 제한 시간과 자식 종료 여유보다 임대가 짧으면 기동을 거절한다."""
        if self.stt_lease_seconds <= self.stt_timeout_seconds + 30:
            raise ValueError("STT_LEASE_SECONDS must exceed STT_TIMEOUT_SECONDS + 30")
        return self


@lru_cache
def get_settings() -> Settings:
    """설정을 한 번만 읽어 재사용한다."""
    return Settings()
