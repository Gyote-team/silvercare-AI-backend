"""채팅용 OpenAI Responses API adapter 계약을 검증한다."""

from types import SimpleNamespace
from uuid import uuid4

from pydantic import SecretStr

from app.contracts.ai_result_document import AnswerSentence, GeneratedAnswer
from app.core.config import Settings
from app.infrastructure.providers.llm.openai_compatible_client import OpenAiCompatibleLlmClient


def test_generate_grounded_answer_uses_responses_api(monkeypatch) -> None:
    """② 채팅 키와 구조화 출력 schema로 Responses API를 호출한다."""
    settings = Settings(
        database_url="postgresql+psycopg://test:test@localhost/test",
        ai_internal_token=SecretStr("test-token"),
        storage_local_root=".",
        llm_api_key=SecretStr("chat-key"),
        llm_model="gpt-test",
        document_analysis_resume_on_startup=False,
    )
    client = OpenAiCompatibleLlmClient(settings)
    citation_id = uuid4()
    captured = {}

    expected_answer = GeneratedAnswer(
        title="검사 결과 설명",
        answer="HbA1c 6.5%가 기록되어 있습니다.",
        sentences=[
            AnswerSentence(
                text="HbA1c 6.5%가 기록되어 있습니다.",
                citation_ids=[citation_id],
            )
        ],
    )

    def parse(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_parsed=expected_answer, usage=None)

    monkeypatch.setattr(client._client.responses, "parse", parse)
    try:
        answer, usage = client.generate_grounded_answer("시스템 지시", "사용자 질문")
    finally:
        client.close()

    assert answer.sections[0].items[0].citation_ids == [citation_id]
    assert usage.total_tokens == 0
    assert captured["model"] == "gpt-test"
    assert captured["instructions"] == "시스템 지시"
    assert captured["input"] == "사용자 질문"
    assert captured["store"] is False
    assert captured["text_format"] is GeneratedAnswer
