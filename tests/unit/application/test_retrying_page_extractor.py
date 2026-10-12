"""재시도 추출기 단위 테스트. 안쪽 포트와 대기 함수는 가짜로 대신해 실제로 기다리지 않는다."""

from collections.abc import Sequence

import pytest

from app.application.ports.page_extractor import (
    PageExtractionError,
    PageExtractionErrorCode,
    PageExtractionResult,
    VlmUsage,
)
from app.application.services.retrying_page_extractor import RetryingPageExtractor
from app.domain.document.page_extraction import DocumentType, PageExtraction
from app.domain.document.page_split import PageImage

PAGE = PageImage(page_no=1, png_bytes=b"page1", width_px=100, height_px=200)
RESULT = PageExtractionResult(
    extraction=PageExtraction(document_type=DocumentType.UNKNOWN, items=[]),
    usage=VlmUsage(
        model_name="fake-model",
        prompt_version="v1",
        input_tokens=10,
        output_tokens=5,
        reasoning_tokens=2,
    ),
)
RETRYABLE_CODES = [
    PageExtractionErrorCode.TIMEOUT,
    PageExtractionErrorCode.RATE_LIMITED,
    PageExtractionErrorCode.API_ERROR,
]
OTHER_CODES = [code for code in PageExtractionErrorCode if code not in RETRYABLE_CODES]


class FakeExtractor:
    """정해 둔 오류 코드를 순서대로 던지고, 다 던진 뒤에는 결과를 돌려주는 가짜 포트."""

    def __init__(self, codes: Sequence[PageExtractionErrorCode]) -> None:
        self.codes = list(codes)
        self.call_count = 0

    def extract_page(self, page: PageImage) -> PageExtractionResult:
        """호출 횟수를 세고, 남은 오류 코드가 있으면 그 코드로 실패한다."""
        self.call_count += 1
        if self.codes:
            raise PageExtractionError(self.codes.pop(0))
        return RESULT


def make_extractor(
    inner: FakeExtractor, waits: list[float], max_retries: int = 2
) -> RetryingPageExtractor:
    """대기 시간을 waits에 기록만 하는 재시도 추출기를 만든다."""
    return RetryingPageExtractor(
        inner, max_retries=max_retries, backoff_seconds=1.0, sleep=waits.append
    )


@pytest.mark.parametrize("code", RETRYABLE_CODES)
def test_retryable_code_is_retried_up_to_limit(code: PageExtractionErrorCode) -> None:
    """재시도 대상 코드는 처음 호출 포함 3번 부르고, 1초·2초를 기다린 뒤 마지막 오류를 올린다."""
    inner = FakeExtractor([code] * 5)
    waits: list[float] = []

    with pytest.raises(PageExtractionError) as error:
        make_extractor(inner, waits).extract_page(PAGE)

    assert error.value.code is code
    assert inner.call_count == 3
    assert waits == [1.0, 2.0]


@pytest.mark.parametrize("code", OTHER_CODES)
def test_other_code_is_called_once(code: PageExtractionErrorCode) -> None:
    """재시도 대상이 아닌 코드는 한 번만 부르고 기다리지 않는다."""
    inner = FakeExtractor([code] * 5)
    waits: list[float] = []

    with pytest.raises(PageExtractionError) as error:
        make_extractor(inner, waits).extract_page(PAGE)

    assert error.value.code is code
    assert inner.call_count == 1
    assert waits == []


def test_returns_result_when_retry_succeeds() -> None:
    """재시도 중에 성공하면 그 결과를 돌려주고 더 부르지 않는다."""
    inner = FakeExtractor([PageExtractionErrorCode.TIMEOUT])
    waits: list[float] = []

    result = make_extractor(inner, waits).extract_page(PAGE)

    assert result is RESULT
    assert inner.call_count == 2
    assert waits == [1.0]


def test_zero_max_retries_calls_once() -> None:
    """max_retries가 0이면 재시도 대상 코드도 한 번만 부른다."""
    inner = FakeExtractor([PageExtractionErrorCode.TIMEOUT])
    waits: list[float] = []

    with pytest.raises(PageExtractionError):
        make_extractor(inner, waits, max_retries=0).extract_page(PAGE)

    assert inner.call_count == 1
    assert waits == []
