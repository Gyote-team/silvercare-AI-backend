"""여러 테스트 폴더가 함께 쓰는 픽스처."""

import pytest
from pydantic import SecretStr

from app.core.config import Settings


@pytest.fixture(autouse=True)
def _disable_resume_on_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """lifespan을 실행하는 테스트가 .env의 실제 DB에서 남은 분석을 조회·실행하지 않게 끈다.

    환경 변수가 .env보다 우선이라 Settings()를 직접 만드는 테스트에도 적용된다.
    """
    monkeypatch.setenv("DOCUMENT_ANALYSIS_RESUME_ON_STARTUP", "false")


@pytest.fixture
def test_settings() -> Settings:
    """외부 키 없이 내부 API 계약만 검증할 수 있는 설정을 만든다."""
    return Settings(
        database_url="postgresql+psycopg://localhost/test",
        ai_internal_token=SecretStr("test-token"),
        storage_local_root="./data/object-storage",
        llm_api_key=SecretStr("test-key"),
        llm_base_url="https://example.test/openai/",
        llm_model="test-model",
    )
