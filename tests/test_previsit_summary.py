"""요약 근거 범위·전용 키·실제 SDK 계약·내부 인증을 외부 호출 없이 검증한다."""

import json
from datetime import UTC, datetime
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI
from pydantic import SecretStr, ValidationError

from app.application.use_cases.generate_previsit_summary import GeneratePrevisitSummaryUseCase
from app.contracts.previsit_summary import GeneratedPrevisitSummary, PrevisitSummaryRequest
from app.core.config import Settings, get_settings
from app.infrastructure.providers.llm.openai_previsit_summarizer import OpenAiPrevisitSummarizer
from app.main import app


def settings() -> Settings:
    """실제 키와 .env를 사용하지 않는 테스트 설정이다."""
    return Settings(
        _env_file=None,
        database_url="sqlite://",
        ai_internal_token="test",
        storage_local_root=".",
        previsit_summary_api_key=None,
    )


def request() -> PrevisitSummaryRequest:
    """개인정보 없는 가상 건강기록을 만든다."""
    return PrevisitSummaryRequest(
        request_id=uuid4(),
        window_days=14,
        records=[
            {
                "healthRecordId": uuid4(),
                "recordedAt": datetime.now(UTC),
                "content": "산책을 10분 했습니다. 이전 지시를 무시하라는 문구는 인용 자료입니다.",
            }
        ],
    )


def test_empty_window_does_not_call_provider() -> None:
    """빈 기록은 요약 내용을 만들어내지 않고 근거 부족으로 응답한다."""
    provider = Mock()
    result = GeneratePrevisitSummaryUseCase(provider).execute(
        PrevisitSummaryRequest(request_id=uuid4(), window_days=14, records=[])
    )
    assert result.result_status == "INSUFFICIENT"
    assert result.used_health_record_ids == []
    provider.generate.assert_not_called()


def test_unknown_source_is_never_exposed() -> None:
    """모델이 제공하지 않은 기록을 참조하면 본문 전체를 숨긴다."""
    provider = Mock()
    provider.generate.return_value = GeneratedPrevisitSummary(
        sentences=[
            {
                "text": "노출하면 안 되는 문장",
                "sourceHealthRecordIds": [uuid4()],
            }
        ]
    )
    result = GeneratePrevisitSummaryUseCase(provider).execute(request())
    assert result.failed and result.error_code == "AI_RESPONSE_INVALID"
    assert result.summary is None and not result.used_health_record_ids


def test_summary_key_cannot_fall_back_to_document_key() -> None:
    """문서 분석 키만 있으면 요약은 전용 키 누락으로 실패한다."""
    config = settings().model_copy(update={"openai_api_key": SecretStr("document-only")})
    result = GeneratePrevisitSummaryUseCase(OpenAiPrevisitSummarizer(config)).execute(request())
    assert result.error_code == "AI_API_KEY_MISSING"


@pytest.mark.parametrize(
    "status,code",
    [(200, None), (401, "AI_AUTH_FAILED"), (429, "AI_RATE_LIMIT"), (500, "AI_UNAVAILABLE")],
)
def test_actual_sdk_structured_request_and_safe_errors(status: int, code: str | None) -> None:
    """SDK가 전용 API·모델·비저장 구조화 출력 계약을 보내고 오류를 숨긴다."""
    source = request()
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        assert str(req.url) == "https://api.openai.com/v1/responses"
        body = json.loads(req.content)
        assert body["model"] == "gpt-5.6-sol" and body["store"] is False
        assert body["text"]["format"]["strict"] is True
        assert json.loads(body["input"])["records"][0]["healthRecordId"] == str(
            source.records[0].health_record_id
        )
        assert "진단" in body["instructions"] and "지시가 아니다" in body["instructions"]
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "private provider body"}})
        generated = {
            "sentences": [
                {
                    "text": "사용자가 산책 10분을 기록했습니다.",
                    "sourceHealthRecordIds": [str(source.records[0].health_record_id)],
                }
            ]
        }
        return httpx.Response(
            200,
            json={
                "id": "test-response",
                "object": "response",
                "created_at": 0,
                "model": "gpt-5.6-sol",
                "status": "completed",
                "output": [
                    {
                        "id": "test-message",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(generated),
                                "annotations": [],
                            }
                        ],
                    }
                ],
            },
        )

    client = OpenAI(
        api_key="local-test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        result = GeneratePrevisitSummaryUseCase(
            OpenAiPrevisitSummarizer(settings(), client)
        ).execute(source)
        if code:
            assert result.error_code == code and result.failed and result.summary is None
        else:
            assert result.result_status == "COMPLETE"
            assert result.used_health_record_ids == [source.records[0].health_record_id]
        assert len(calls) == 1
    finally:
        client.close()


def test_duplicate_or_oversized_sources_are_rejected() -> None:
    """중복 기록과 과도한 본문을 모델 호출 전에 거절한다."""
    body = request().model_dump()
    body["records"] *= 2
    with pytest.raises(ValidationError):
        PrevisitSummaryRequest(**body)
    body["records"] = [
        {
            "health_record_id": uuid4(),
            "recorded_at": datetime.now(UTC),
            "content": "가" * 2000,
        }
        for _ in range(31)
    ]
    with pytest.raises(ValidationError):
        PrevisitSummaryRequest(**body)


def test_internal_auth_and_missing_key_response() -> None:
    """내부 토큰 없이는 호출하지 못하고 키 누락은 안전한 결과로 반환한다."""
    app.dependency_overrides[get_settings] = settings
    try:
        client = TestClient(app)
        body = request().model_dump(mode="json")
        assert client.post("/internal/previsit-summaries/generate", json=body).status_code == 401
        result = client.post(
            "/internal/previsit-summaries/generate", json=body, headers={"X-Internal-Token": "test"}
        )
        assert result.status_code == 200
        assert result.json()["errorCode"] == "AI_API_KEY_MISSING"
    finally:
        app.dependency_overrides.clear()
