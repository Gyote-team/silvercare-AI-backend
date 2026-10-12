"""OpenAI 페이지 추출기 테스트. httpx.MockTransport로 대신해 실제 API를 부르지 않는다."""

import base64
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from app.application.ports.page_extractor import (
    PageExtractionError,
    PageExtractionErrorCode,
    VlmUsage,
)
from app.core.config import Settings
from app.domain.document.page_extraction import DocumentType, ExtractedItemType, SourceBox
from app.domain.document.page_split import PageImage
from app.infrastructure.providers.llm.openai_page_extractor import (
    OpenAIPageExtractor,
    build_page_extractor,
)

API_KEY = "sk-test-secret-value"
SECRET_API_KEY = SecretStr(API_KEY)
SOURCE_TEXT = "혈색소 13.5 g/dL"
PAGE = PageImage(page_no=1, png_bytes=b"fake-png-bytes", width_px=1240, height_px=1754)
SETTINGS_ENV_NAMES = [
    "OPENAI_API_KEY",
    "OPENAI_VLM_MODEL",
    "OPENAI_VLM_REASONING_EFFORT",
    "OPENAI_VLM_TIMEOUT_SECONDS",
]

Handler = Callable[[httpx.Request], httpx.Response]


def make_output_text(**item_overrides: Any) -> str:
    """합성 검사값 항목 하나가 든 모델 출력 JSON 문자열을 만든다. 박스는 0~1000 좌표다."""
    item: dict[str, Any] = {
        "item_type": "LAB_VALUE",
        "normalized_name": "혈색소",
        "raw_value": "13.5",
        "normalized_value": "13.5",
        "unit": "g/dL",
        "reference_range": "12.0-16.0",
        "confidence": 0.9,
        "source_text": SOURCE_TEXT,
        "source_box": {"x": 100, "y": 200, "width": 500, "height": 100},
    }
    page = {"document_type": "LAB_RESULT", "items": [{**item, **item_overrides}]}
    return json.dumps(page, ensure_ascii=False)


def make_body(content: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Responses API 응답 본문을 만든다. 추론 항목 뒤에 메시지 항목이 온다."""
    body: dict[str, Any] = {
        "status": "completed",
        "model": "gpt-5.6-sol-2026-09-01",
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [content]},
        ],
        "usage": {
            "input_tokens": 4000,
            "output_tokens": 3000,
            "output_tokens_details": {"reasoning_tokens": 2500},
        },
    }
    return {**body, **overrides}


def text_body(text: str, **overrides: Any) -> dict[str, Any]:
    """모델 출력 글자가 든 응답 본문을 만든다."""
    return make_body({"type": "output_text", "text": text}, **overrides)


class Recorder:
    """받은 요청을 기록하고 정해 둔 응답을 돌려주는 가짜 HTTP 처리기."""

    def __init__(self, status_code: int = 200, body: Any = None) -> None:
        self.status_code = status_code
        self.body = body
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """요청을 기록하고 응답을 만든다."""
        self.requests.append(request)
        return httpx.Response(self.status_code, json=self.body)


def make_extractor(
    handler: Handler, api_key: SecretStr | None = SECRET_API_KEY
) -> OpenAIPageExtractor:
    """가짜 HTTP 처리기를 쓰는 추출기를 만든다."""
    return OpenAIPageExtractor(
        api_key=api_key,
        model="gpt-5.6-sol",
        reasoning_effort="high",
        timeout_seconds=5,
        prompt="PROMPT",
        prompt_version="v1",
        transport=httpx.MockTransport(handler),
    )


def extract_error_code(handler: Handler) -> PageExtractionErrorCode:
    """추출을 실행해 올라온 PageExtractionError의 코드를 돌려준다."""
    with pytest.raises(PageExtractionError) as error:
        make_extractor(handler).extract_page(PAGE)
    return error.value.code


def test_sends_image_prompt_and_strict_schema() -> None:
    """모델, 추론 강도, 프롬프트, 이미지, strict 스키마, 키를 요청에 담는다."""
    recorder = Recorder(body=text_body(make_output_text()))

    make_extractor(recorder).extract_page(PAGE)

    request = recorder.requests[0]
    payload = json.loads(request.content)
    image = payload["input"][0]["content"][0]
    output_format = payload["text"]["format"]
    assert len(recorder.requests) == 1
    assert str(request.url) == "https://api.openai.com/v1/responses"
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert payload["model"] == "gpt-5.6-sol"
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["instructions"] == "PROMPT"
    assert payload["store"] is False
    assert image["type"] == "input_image"
    assert image["image_url"] == (
        "data:image/png;base64," + base64.b64encode(b"fake-png-bytes").decode("ascii")
    )
    assert (output_format["type"], output_format["strict"]) == ("json_schema", True)
    assert set(output_format["schema"]["properties"]) == {"document_type", "items"}


def test_returns_extraction_with_pixel_box_and_usage() -> None:
    """정상 응답은 추출 결과로 읽고, 박스는 픽셀로 바꾸고, 사용량과 모델 이름을 돌려준다."""
    recorder = Recorder(body=text_body(make_output_text()))

    result = make_extractor(recorder).extract_page(PAGE)

    item = result.extraction.items[0]
    assert result.extraction.document_type is DocumentType.LAB_RESULT
    assert (item.item_type, item.source_text) == (ExtractedItemType.LAB_VALUE, SOURCE_TEXT)
    assert item.source_box == SourceBox(x=124, y=350, width=620, height=176)
    assert result.usage == VlmUsage(
        model_name="gpt-5.6-sol-2026-09-01",
        prompt_version="v1",
        input_tokens=4000,
        output_tokens=3000,
        reasoning_tokens=2500,
    )


def test_box_outside_page_is_dropped_but_item_is_kept() -> None:
    """페이지를 벗어난 박스는 None으로 바꾸고 항목은 그대로 둔다."""
    box = {"x": 900, "y": 0, "width": 200, "height": 10}
    recorder = Recorder(body=text_body(make_output_text(source_box=box)))

    result = make_extractor(recorder).extract_page(PAGE)

    assert result.extraction.items[0].source_box is None
    assert result.extraction.items[0].source_text == SOURCE_TEXT


@pytest.mark.parametrize(
    "body",
    [
        text_body("not json"),
        text_body(make_output_text(source_text="")),
        text_body(make_output_text(item_type="SURGERY")),
        text_body(make_output_text(), output=[]),
        {"status": "completed"},
        ["not", "an", "object"],
    ],
)
def test_schema_mismatch(body: Any) -> None:
    """JSON이 아니거나 출력 형식과 다르거나 출력이 없으면 SCHEMA_MISMATCH다."""
    assert extract_error_code(Recorder(body=body)) is PageExtractionErrorCode.SCHEMA_MISMATCH


def test_refusal() -> None:
    """모델이 거절하면 REFUSED다."""
    body = make_body({"type": "refusal", "refusal": "거절합니다"})

    assert extract_error_code(Recorder(body=body)) is PageExtractionErrorCode.REFUSED


def test_incomplete_response() -> None:
    """응답 상태가 completed가 아니면 INCOMPLETE다."""
    body = text_body("", status="incomplete")

    assert extract_error_code(Recorder(body=body)) is PageExtractionErrorCode.INCOMPLETE


def test_timeout() -> None:
    """제한 시간을 넘기면 TIMEOUT이다."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    assert extract_error_code(handler) is PageExtractionErrorCode.TIMEOUT


def test_network_error() -> None:
    """연결하지 못하면 API_ERROR다."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    assert extract_error_code(handler) is PageExtractionErrorCode.API_ERROR


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (429, PageExtractionErrorCode.RATE_LIMITED),
        (500, PageExtractionErrorCode.API_ERROR),
        (503, PageExtractionErrorCode.API_ERROR),
        (400, PageExtractionErrorCode.REQUEST_REJECTED),
        (401, PageExtractionErrorCode.REQUEST_REJECTED),
    ],
)
def test_http_error_status(status_code: int, expected: PageExtractionErrorCode) -> None:
    """HTTP 오류 상태를 우리 오류 코드로 바꾼다."""
    recorder = Recorder(status_code=status_code, body={"error": {"message": "failed"}})

    assert extract_error_code(recorder) is expected


@pytest.mark.parametrize("api_key", [None, SecretStr("")])
def test_missing_api_key_makes_no_http_call(api_key: SecretStr | None) -> None:
    """키가 없거나 비어 있으면 HTTP 호출 없이 MISSING_API_KEY다."""
    recorder = Recorder(body=text_body(make_output_text()))

    with pytest.raises(PageExtractionError) as error:
        make_extractor(recorder, api_key).extract_page(PAGE)

    assert error.value.code is PageExtractionErrorCode.MISSING_API_KEY
    assert recorder.requests == []


@pytest.mark.parametrize(
    ("status_code", "body"),
    [
        (200, text_body(make_output_text(item_type="SURGERY"))),
        (401, {"error": {"message": f"Incorrect API key provided: {API_KEY}"}}),
    ],
)
def test_error_hides_key_and_source_text(status_code: int, body: dict[str, Any]) -> None:
    """예외에는 오류 코드만 있고 키, 문서 원문, 원인 예외가 없다."""
    with pytest.raises(PageExtractionError) as error:
        make_extractor(Recorder(status_code, body)).extract_page(PAGE)

    assert str(error.value) == error.value.code.value
    assert API_KEY not in repr(error.value)
    assert SOURCE_TEXT not in repr(error.value)
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__ or error.value.__context__ is None


def test_close_closes_http_client() -> None:
    """close 뒤에는 HTTP 호출을 하지 못한다."""
    recorder = Recorder(body=text_body(make_output_text()))
    extractor = make_extractor(recorder)

    extractor.close()

    with pytest.raises(RuntimeError):
        extractor.extract_page(PAGE)
    assert recorder.requests == []


def test_build_from_default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """설정 기본값과 v1 프롬프트로 만든다. 키가 없어도 만들어지고 호출할 때만 실패한다."""
    for name in SETTINGS_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    settings = Settings(
        _env_file=None, database_url="unused", ai_internal_token="token", storage_local_root="root"
    )
    recorder = Recorder(body=text_body(make_output_text()))

    with pytest.raises(PageExtractionError) as error:
        build_page_extractor(settings, httpx.MockTransport(recorder)).extract_page(PAGE)

    assert error.value.code is PageExtractionErrorCode.MISSING_API_KEY
    assert (settings.openai_vlm_model, settings.openai_vlm_reasoning_effort) == (
        "gpt-5.6-sol",
        "high",
    )
    assert settings.openai_vlm_timeout_seconds == 180


def test_build_uses_settings_and_prompt_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """설정의 키·모델·추론 강도와 프롬프트 파일 내용을 요청에 쓴다."""
    for name in SETTINGS_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    settings = Settings(
        _env_file=None,
        database_url="unused",
        ai_internal_token="token",
        storage_local_root="root",
        openai_api_key=API_KEY,
        openai_vlm_reasoning_effort="low",
    )
    recorder = Recorder(body=text_body(make_output_text()))

    result = build_page_extractor(settings, httpx.MockTransport(recorder)).extract_page(PAGE)

    payload = json.loads(recorder.requests[0].content)
    assert recorder.requests[0].headers["Authorization"] == f"Bearer {API_KEY}"
    assert (payload["model"], payload["reasoning"]) == ("gpt-5.6-sol", {"effort": "low"})
    assert payload["instructions"].startswith("# 의료 문서 페이지 추출 v1")
    assert result.usage.prompt_version == "v1"
