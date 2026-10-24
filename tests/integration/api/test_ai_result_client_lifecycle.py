"""LLM 클라이언트의 앱 생명주기 재사용을 검증한다."""

from uuid import uuid4

from fastapi.testclient import TestClient

from app.contracts.ai_result_document import AnswerSentence, GeneratedAnswer
from app.core.config import get_settings
from app.main import app
from tests.integration.api.test_error_responses import _settings
from tests.unit.application.test_generate_result_document import request_with_evidence


def test_llm_client_is_reused_and_closed(monkeypatch) -> None:
    """두 요청이 같은 클라이언트를 사용하고 앱 종료 때 한 번 닫는다."""
    instances = []

    class FakeClient:
        def __init__(self, settings) -> None:
            self.closed = False
            self.calls = 0
            instances.append(self)

        def generate_grounded_answer(self, system_prompt, user_prompt):
            self.calls += 1
            return GeneratedAnswer(
                title="설명",
                answer="근거 있는 문장",
                sentences=[AnswerSentence(text="근거 있는 문장", citation_ids=[citation_id])],
            )

        def close(self) -> None:
            self.closed = True

    citation_id = uuid4()
    monkeypatch.setattr("app.main.OpenAiCompatibleLlmClient", FakeClient)
    monkeypatch.setattr("app.main.get_settings", _settings)
    app.dependency_overrides[get_settings] = _settings
    try:
        with TestClient(app) as client:
            for _ in range(2):
                response = client.post(
                    "/internal/ai-result-documents/generate",
                    headers={"X-Internal-Token": "test-token"},
                    json=request_with_evidence(citation_id).model_dump(mode="json"),
                )
                assert response.status_code == 200
                assert response.json()["result_status"] == "COMPLETE"
            assert len(instances) == 1
            assert instances[0].calls == 2
            assert instances[0].closed is False
    finally:
        app.dependency_overrides.clear()

    assert instances[0].closed is True
