"""채팅 키가 없어도 요약 API가 기동하고 다른 기능 키를 사용하지 않는지 확인한다."""

from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import Settings, get_settings
from app.domain.ai_result.error import AiResultErrorCode, AiResultGenerationError
from app.infrastructure.providers.llm.openai_compatible_client import OpenAiCompatibleLlmClient
from app.main import app


@pytest.mark.parametrize("key", [None, SecretStr(""), SecretStr("   ")])
def test_missing_chat_key_does_not_create_provider_or_fall_back(
    key: SecretStr | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """빈 채팅 키는 다른 키로 대체하지 않고 호출 시 안전한 오류를 반환한다."""
    settings = Settings(
        _env_file=None,
        database_url="sqlite://",
        storage_local_root=".",
        ai_internal_token="test",
        llm_api_key=key,
        openai_api_key="document-only",
    )
    factory = Mock()
    monkeypatch.setattr("app.infrastructure.providers.llm.openai_compatible_client.OpenAI", factory)
    client = OpenAiCompatibleLlmClient(settings)
    with pytest.raises(AiResultGenerationError) as error:
        client.generate_grounded_answer("system", "user")
    assert error.value.code == AiResultErrorCode.AI_API_KEY_MISSING
    factory.assert_not_called()
    client.close()


def test_previsit_route_starts_without_any_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """실제 lifespan에서 키 없이 기동해 빈 기록을 처리하며 유료 호출을 하지 않는다."""
    settings = Settings(
        _env_file=None,
        database_url="sqlite://",
        storage_local_root=".",
        ai_internal_token="test",
        llm_api_key=None,
        previsit_summary_api_key=None,
        document_analysis_resume_on_startup=False,
    )
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            response = client.post(
                "/internal/previsit-summaries/generate",
                headers={"X-Internal-Token": "test"},
                json={"requestId": str(uuid4()), "windowDays": 14, "records": []},
            )
            assert response.status_code == 200
            assert response.json()["resultStatus"] == "INSUFFICIENT"
    finally:
        app.dependency_overrides.clear()
