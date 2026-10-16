"""내부 AI API의 공통 오류 응답 계약을 검증한다."""

from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import Settings, get_settings
from app.main import app


def _settings() -> Settings:
    """외부 키 없이 내부 API 계약만 검증할 수 있는 설정을 만든다."""
    return Settings(
        database_url="postgresql+psycopg://localhost/test",
        ai_internal_token=SecretStr("test-token"),
        storage_local_root="./data/object-storage",
        llm_api_key=SecretStr("test-key"),
        llm_base_url="https://example.test/openai/",
        llm_model="test-model",
    )


def test_invalid_request_uses_shared_error_shape() -> None:
    """잘못된 요청 본문은 GLOBAL_002 공통 JSON으로 반환한다."""
    app.dependency_overrides[get_settings] = _settings
    try:
        response = TestClient(app).post(
            "/internal/ai-result-documents/generate",
            headers={"X-Internal-Token": "test-token"},
            json={},
        )
    finally:
        app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 400
    assert body["code"] == "GLOBAL_002"
    assert body["message"] == "요청 값이 올바르지 않습니다."
    assert "timestamp" in body


def test_invalid_internal_token_uses_shared_error_shape() -> None:
    """잘못된 내부 토큰은 AUTH_001 공통 JSON으로 반환한다."""
    app.dependency_overrides[get_settings] = _settings
    try:
        response = TestClient(app).post(
            "/internal/ai-result-documents/generate",
            headers={"X-Internal-Token": "wrong-token"},
            json={},
        )
    finally:
        app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 401
    assert body["code"] == "AUTH_001"
    assert body["message"] == "인증 정보가 올바르지 않습니다."
    assert "timestamp" in body
