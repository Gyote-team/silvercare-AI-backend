"""여러 테스트 폴더가 함께 쓰는 픽스처."""

import pytest
from pydantic import SecretStr

from app.core.config import Settings


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
