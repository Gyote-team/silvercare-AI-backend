"""서버 설정의 기본값을 검증한다."""

import pytest
from pydantic import SecretStr

from app.core.config import Settings


def test_embedding_key_does_not_fall_back_to_other_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPENAI_API_KEY나 LLM_API_KEY만 있으면 임베딩 키는 비어 있다."""
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("LLM_API_KEY", "sk-llm")

    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://localhost/test",
        ai_internal_token=SecretStr("test-token"),
        storage_local_root="./data",
    )

    assert settings.embedding_api_key is None
    assert settings.embedding_model == "text-embedding-3-small"


def _default(name: str) -> object:
    """.env와 환경 변수를 읽지 않고 필드 기본값만 꺼낸다."""
    return Settings.model_fields[name].default


def test_stale_minutes_default_exceeds_worst_analysis_time() -> None:
    """멈춘 분석 판단 시간의 기본값이 기본 설정의 최악 분석 시간보다 길다."""
    attempts = _default("openai_vlm_max_retries") + 1
    per_attempt = _default("openai_vlm_connect_timeout_seconds") + _default(
        "openai_vlm_timeout_seconds"
    )
    backoff = _default("openai_vlm_retry_backoff_seconds") * (
        2 ** _default("openai_vlm_max_retries") - 1
    )
    worst_seconds = _default("document_max_pages") * (attempts * per_attempt + backoff)

    assert _default("document_analysis_stale_minutes") * 60 > worst_seconds
