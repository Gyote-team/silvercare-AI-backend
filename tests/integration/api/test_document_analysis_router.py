"""문서 분석 요청 내부 API의 응답 코드와 백그라운드 등록을 검증한다."""

from collections.abc import Callable
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.internal.v1 import document_analysis_router
from app.core.config import Settings, get_settings
from app.main import app

_REQUEST_ID = "req-test-1"
_OBJECT_KEY = "documents/synthetic-test.pdf"
_MIME_TYPE = "application/pdf"


def _body(document_id: UUID) -> dict[str, str]:
    """Spring 계약대로 camelCase인 합성 요청 본문을 만든다."""
    return {
        "documentId": str(document_id),
        "objectKey": _OBJECT_KEY,
        "mimeType": _MIME_TYPE,
        "requestId": _REQUEST_ID,
    }


@pytest.fixture
def post(test_settings: Settings) -> Callable[..., httpx.Response]:
    """테스트 설정으로 분석 요청 API를 호출하는 함수를 돌려준다. lifespan은 실행하지 않는다."""

    def _post(
        path_id: object, body: dict[str, str], token: str | None = "test-token"
    ) -> httpx.Response:
        headers = {"X-Internal-Token": token} if token is not None else {}
        app.dependency_overrides[get_settings] = lambda: test_settings
        try:
            return TestClient(app).post(
                f"/internal/documents/{path_id}/analyze", headers=headers, json=body
            )
        finally:
            app.dependency_overrides.clear()

    return _post


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


def test_valid_request_returns_202_and_schedules_analysis_once(calls, post) -> None:
    """정상 요청은 본문 없는 202를 반환하고 요청 값으로 분석을 한 번 등록한다."""
    document_id = uuid4()

    response = post(document_id, _body(document_id))

    assert response.status_code == 202
    assert response.content == b""
    assert calls == [(document_id, _REQUEST_ID, _OBJECT_KEY, _MIME_TYPE)]


def test_snake_case_body_is_accepted(calls, post) -> None:
    """snake_case 필드 이름으로 보내도 202로 받는다."""
    document_id = uuid4()
    body = {
        "document_id": str(document_id),
        "object_key": _OBJECT_KEY,
        "mime_type": _MIME_TYPE,
        "request_id": _REQUEST_ID,
    }

    assert post(document_id, body).status_code == 202
    assert calls == [(document_id, _REQUEST_ID, _OBJECT_KEY, _MIME_TYPE)]


def test_missing_token_is_rejected_with_401(calls, post) -> None:
    """내부 토큰 헤더가 없으면 401 AUTH_001이다."""
    document_id = uuid4()

    _assert_error(post(document_id, _body(document_id), token=None), 401, "AUTH_001")
    assert calls == []


def test_wrong_token_is_rejected_with_401(calls, post) -> None:
    """내부 토큰이 다르면 401 AUTH_001이다."""
    document_id = uuid4()

    _assert_error(post(document_id, _body(document_id), token="wrong-token"), 401, "AUTH_001")
    assert calls == []


def test_missing_body_field_is_rejected_with_400(calls, post) -> None:
    """body 필드가 빠지면 400 GLOBAL_002이다."""
    document_id = uuid4()
    body = _body(document_id)
    del body["objectKey"]

    _assert_error(post(document_id, body), 400, "GLOBAL_002")
    assert calls == []


@pytest.mark.parametrize("request_id", ["a" * 65, "req-1\nforged", "req 1"])
def test_invalid_request_id_is_rejected_with_400(calls, post, request_id: str) -> None:
    """requestId가 64자를 넘거나 허용하지 않는 문자(개행, 공백)를 담으면 400 GLOBAL_002이다."""
    document_id = uuid4()
    body = _body(document_id) | {"requestId": request_id}

    _assert_error(post(document_id, body), 400, "GLOBAL_002")
    assert calls == []


def test_request_id_of_64_chars_is_accepted(calls, post) -> None:
    """Spring 컬럼 길이와 같은 64자 requestId는 받는다."""
    document_id = uuid4()
    body = _body(document_id) | {"requestId": "a" * 64}

    assert post(document_id, body).status_code == 202


def test_path_and_body_document_id_mismatch_is_rejected_with_400(calls, post) -> None:
    """경로와 body의 documentId가 다르면 400 GLOBAL_002이고 분석을 등록하지 않는다."""
    _assert_error(post(uuid4(), _body(uuid4())), 400, "GLOBAL_002")
    assert calls == []


def test_non_uuid_path_document_id_is_rejected_with_400(calls, post) -> None:
    """경로 documentId가 UUID 형식이 아니면 400 GLOBAL_002이고 분석을 등록하지 않는다."""
    _assert_error(post("not-a-uuid", _body(uuid4())), 400, "GLOBAL_002")
    assert calls == []
