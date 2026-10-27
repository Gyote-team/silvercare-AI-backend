"""문서 분석 요청 내부 API의 응답 코드와 백그라운드 등록을 검증한다."""

import logging
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.internal.v1 import document_analysis_router
from app.core.config import get_settings
from app.main import app
from app.workers import document_analysis_task
from tests.integration.api.test_error_responses import _settings

_REQUEST_ID = "req-test-1"
_OBJECT_KEY = "documents/synthetic-test.pdf"


def _body(document_id: UUID) -> dict[str, str]:
    """Spring 계약대로 camelCase인 합성 요청 본문을 만든다."""
    return {
        "documentId": str(document_id),
        "objectKey": _OBJECT_KEY,
        "mimeType": "application/pdf",
        "requestId": _REQUEST_ID,
    }


def _post(
    path_id: object, body: dict[str, str], token: str | None = "test-token"
) -> httpx.Response:
    """테스트 설정으로 분석 요청 API를 한 번 호출한다. lifespan은 실행하지 않는다."""
    headers = {"X-Internal-Token": token} if token is not None else {}
    app.dependency_overrides[get_settings] = _settings
    try:
        return TestClient(app).post(
            f"/internal/documents/{path_id}/analyze", headers=headers, json=body
        )
    finally:
        app.dependency_overrides.clear()


def _assert_error(response: httpx.Response, status_code: int, code: str) -> None:
    """공통 오류 JSON의 상태 코드와 오류 코드를 확인한다."""
    assert response.status_code == status_code
    assert response.json()["code"] == code


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, ...]]:
    """실제 분석 대신 호출 인자만 기록하는 가짜 실행 함수를 넣는다."""
    recorded: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        document_analysis_router, "run_document_analysis", lambda *args: recorded.append(args)
    )
    return recorded


def test_valid_request_returns_202_and_schedules_analysis_once(calls) -> None:
    """정상 요청은 202를 반환하고 documentId와 requestId로 분석을 한 번 등록한다."""
    document_id = uuid4()

    response = _post(document_id, _body(document_id))

    assert response.status_code == 202
    assert calls == [(document_id, _REQUEST_ID)]


def test_missing_token_is_rejected_with_401(calls) -> None:
    """내부 토큰 헤더가 없으면 401 AUTH_001이다."""
    document_id = uuid4()

    _assert_error(_post(document_id, _body(document_id), token=None), 401, "AUTH_001")
    assert calls == []


def test_wrong_token_is_rejected_with_401(calls) -> None:
    """내부 토큰이 다르면 401 AUTH_001이다."""
    document_id = uuid4()

    _assert_error(_post(document_id, _body(document_id), token="wrong-token"), 401, "AUTH_001")
    assert calls == []


def test_missing_body_field_is_rejected_with_400(calls) -> None:
    """body 필드가 빠지면 400 GLOBAL_002이다."""
    document_id = uuid4()
    body = _body(document_id)
    del body["objectKey"]

    _assert_error(_post(document_id, body), 400, "GLOBAL_002")
    assert calls == []


def test_path_and_body_document_id_mismatch_is_rejected_with_400(calls) -> None:
    """경로와 body의 documentId가 다르면 400 GLOBAL_002이고 분석을 등록하지 않는다."""
    _assert_error(_post(uuid4(), _body(uuid4())), 400, "GLOBAL_002")
    assert calls == []


def test_non_uuid_path_document_id_is_rejected_with_400(calls) -> None:
    """경로 documentId가 UUID 형식이 아니면 400 GLOBAL_002이고 분석을 등록하지 않는다."""
    _assert_error(_post("not-a-uuid", _body(uuid4())), 400, "GLOBAL_002")
    assert calls == []


def test_background_failure_returns_202_and_logs_ids_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """백그라운드 분석이 예외를 내도 202이고, 로그에는 식별자와 예외 이름만 남는다."""

    def fail(*args: object, **kwargs: object) -> None:
        # 메시지에 objectKey를 넣어 로그로 새지 않는지 함께 확인한다.
        raise RuntimeError(_OBJECT_KEY)

    monkeypatch.setattr(document_analysis_task, "get_settings", _settings)
    monkeypatch.setattr(document_analysis_task, "get_session_factory", lambda: None)
    monkeypatch.setattr(document_analysis_task, "analyze_document", fail)
    document_id = uuid4()

    with caplog.at_level(logging.ERROR, logger=document_analysis_task.__name__):
        response = _post(document_id, _body(document_id))

    assert response.status_code == 202
    assert str(document_id) in caplog.text
    assert _REQUEST_ID in caplog.text
    assert "RuntimeError" in caplog.text
    assert _OBJECT_KEY not in caplog.text
