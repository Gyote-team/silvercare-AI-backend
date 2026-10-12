"""OpenAI Responses API로 페이지 PNG에서 추출 결과를 받는 VLM 포트 구현.

openai SDK 없이 httpx로 직접 부른다. 재시도는 하지 않는다(RetryingPageExtractor가 감싸서 한다).
"""

import base64
from typing import Any

import httpx
from pydantic import SecretStr, ValidationError

from app.application.ports.page_extractor import (
    PageExtractionError,
    PageExtractionErrorCode,
    PageExtractionResult,
    VlmUsage,
)
from app.core.config import Settings
from app.domain.document.document_page import RENDERED_PAGE_MIME_TYPE
from app.domain.document.page_extraction import ExtractedItem, NormalizedBox, PageExtraction
from app.domain.document.page_split import PageImage
from app.infrastructure.providers.llm.prompt_loader import load_prompt

PROMPT_FEATURE = "document_extraction"
PROMPT_VERSION = "v1"

_RESPONSES_URL = "https://api.openai.com/v1/responses"
_CONNECT_TIMEOUT_SECONDS = 10.0
# high는 2048x2048 안, 2500 패치까지 원래 크기를 유지한다. 150DPI A4는 축소되지 않는다.
_IMAGE_DETAIL = "high"
# 추론 토큰을 포함한 출력 상한. 한 번의 호출 비용이 끝없이 커지지 않게 막는다.
_MAX_OUTPUT_TOKENS = 16000
_SCHEMA_NAME = "page_extraction"


class _WireItem(ExtractedItem):
    """VLM 응답의 항목. source_box만 0~1000 정규화 좌표다."""

    source_box: NormalizedBox | None


class _WirePage(PageExtraction):
    """VLM 응답의 페이지 결과. 이 모델의 JSON 스키마를 구조화 출력 형식으로 보낸다."""

    items: list[_WireItem]


class OpenAIPageExtractor:
    """OpenAI에 페이지 이미지를 보내 추출 결과를 받는다.

    API 키는 extract_page를 부를 때만 확인한다.
    """

    def __init__(
        self,
        *,
        api_key: SecretStr | None,
        model: str,
        reasoning_effort: str,
        timeout_seconds: float,
        prompt: str,
        prompt_version: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._prompt = prompt
        self._prompt_version = prompt_version
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds, connect=_CONNECT_TIMEOUT_SECONDS),
            transport=transport,
        )

    def extract_page(self, page: PageImage) -> PageExtractionResult:
        """페이지 한 장을 한 번 호출한다. 실패하면 PageExtractionError를 던진다."""
        api_key = self._api_key.get_secret_value() if self._api_key is not None else ""
        if not api_key:
            raise PageExtractionError(PageExtractionErrorCode.MISSING_API_KEY)
        body = self._post(api_key, self._request_body(page))
        wire_page = _parse_output(body)
        return PageExtractionResult(
            extraction=_to_pixel_page(wire_page, page),
            usage=self._usage(body),
        )

    def close(self) -> None:
        """HTTP 연결을 닫는다. 추출기를 만든 쪽이 프로세스를 끝낼 때 한 번 부른다."""
        self._client.close()

    def _request_body(self, page: PageImage) -> dict[str, Any]:
        image_base64 = base64.b64encode(page.png_bytes).decode("ascii")
        return {
            "model": self._model,
            "instructions": self._prompt,
            "reasoning": {"effort": self._reasoning_effort},
            "max_output_tokens": _MAX_OUTPUT_TOKENS,
            # 의료 문서이므로 OpenAI 쪽에 응답을 저장하지 않게 한다.
            "store": False,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:{RENDERED_PAGE_MIME_TYPE};base64,{image_base64}",
                            "detail": _IMAGE_DETAIL,
                        }
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": _SCHEMA_NAME,
                    "strict": True,
                    "schema": _WirePage.model_json_schema(),
                }
            },
        }

    def _post(self, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        # httpx 예외는 요청 객체(키가 든 헤더, 이미지)를 들고 있으므로 from None으로 끊는다.
        try:
            response = self._client.post(
                _RESPONSES_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
        except httpx.TimeoutException:
            raise PageExtractionError(PageExtractionErrorCode.TIMEOUT) from None
        except httpx.HTTPError:
            raise PageExtractionError(PageExtractionErrorCode.API_ERROR) from None
        if response.status_code == 429:
            raise PageExtractionError(PageExtractionErrorCode.RATE_LIMITED)
        if 400 <= response.status_code < 500:
            raise PageExtractionError(PageExtractionErrorCode.REQUEST_REJECTED)
        if not response.is_success:
            raise PageExtractionError(PageExtractionErrorCode.API_ERROR)
        try:
            body = response.json()
        except ValueError:
            raise PageExtractionError(PageExtractionErrorCode.SCHEMA_MISMATCH) from None
        if not isinstance(body, dict):
            raise PageExtractionError(PageExtractionErrorCode.SCHEMA_MISMATCH)
        return body

    def _usage(self, body: dict[str, Any]) -> VlmUsage:
        usage = body.get("usage") or {}
        details = usage.get("output_tokens_details") or {}
        return VlmUsage(
            # 응답의 model은 날짜가 붙은 실제 모델 이름이다. 없으면 요청한 이름을 쓴다.
            model_name=body.get("model") or self._model,
            prompt_version=self._prompt_version,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            reasoning_tokens=details.get("reasoning_tokens", 0),
        )


def _parse_output(body: dict[str, Any]) -> _WirePage:
    if body.get("status") != "completed":
        raise PageExtractionError(PageExtractionErrorCode.INCOMPLETE)
    # 검증 오류에는 문서 원문이 들어 있으므로 from None으로 끊는다.
    try:
        contents = [
            content
            for item in body["output"]
            if item["type"] == "message"
            for content in item["content"]
        ]
        if any(content["type"] == "refusal" for content in contents):
            raise PageExtractionError(PageExtractionErrorCode.REFUSED)
        text = "".join(c["text"] for c in contents if c["type"] == "output_text")
        return _WirePage.model_validate_json(text)
    except (KeyError, TypeError, ValidationError):
        raise PageExtractionError(PageExtractionErrorCode.SCHEMA_MISMATCH) from None


def _to_pixel_page(wire_page: _WirePage, page: PageImage) -> PageExtraction:
    items = [
        ExtractedItem(
            **item.model_dump(exclude={"source_box"}),
            source_box=(
                item.source_box.to_pixel_box(page.width_px, page.height_px)
                if item.source_box is not None
                else None
            ),
        )
        for item in wire_page.items
    ]
    return PageExtraction(document_type=wire_page.document_type, items=items)


def build_page_extractor(
    settings: Settings, transport: httpx.BaseTransport | None = None
) -> OpenAIPageExtractor:
    """설정과 프롬프트 파일로 추출기를 만든다. 이때는 API 키가 없어도 된다."""
    return OpenAIPageExtractor(
        api_key=settings.openai_api_key,
        model=settings.openai_vlm_model,
        reasoning_effort=settings.openai_vlm_reasoning_effort,
        timeout_seconds=settings.openai_vlm_timeout_seconds,
        prompt=load_prompt(PROMPT_FEATURE, PROMPT_VERSION),
        prompt_version=PROMPT_VERSION,
        transport=transport,
    )
