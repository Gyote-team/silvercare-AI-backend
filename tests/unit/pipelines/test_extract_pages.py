"""페이지별 VLM 호출 단계 단위 테스트. VLM 포트는 가짜로 대신한다."""

import pytest

from app.application.ports.page_extractor import (
    PageExtractionError,
    PageExtractionErrorCode,
    PageExtractionResult,
    VlmUsage,
)
from app.domain.document.page_extraction import DocumentType, PageExtraction
from app.domain.document.page_split import PageImage
from app.pipelines.documents.steps.extract_pages import PageExtractionOutcome, extract_pages

PAGES = [
    PageImage(page_no=1, png_bytes=b"page1", width_px=100, height_px=200),
    PageImage(page_no=2, png_bytes=b"page2", width_px=100, height_px=200),
    PageImage(page_no=3, png_bytes=b"page3", width_px=100, height_px=200),
]
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


class FakeExtractor:
    """받은 페이지 번호를 기록하는 가짜 포트. 정해 둔 페이지에서는 정해 둔 예외를 던진다."""

    def __init__(self, errors: dict[int, Exception] | None = None) -> None:
        self.errors = errors or {}
        self.requested_page_nos: list[int] = []

    def extract_page(self, page: PageImage) -> PageExtractionResult:
        """호출을 기록한다. 실패하도록 정해 둔 페이지면 그 예외를 던진다."""
        self.requested_page_nos.append(page.page_no)
        if page.page_no in self.errors:
            raise self.errors[page.page_no]
        return RESULT


def test_all_pages_succeed_in_order() -> None:
    """페이지마다 한 번씩 부르고, 입력 순서대로 결과를 돌려준다."""
    extractor = FakeExtractor()

    outcomes = extract_pages(extractor, PAGES)

    assert extractor.requested_page_nos == [1, 2, 3]
    assert outcomes == [PageExtractionOutcome(page_no=no, result=RESULT) for no in (1, 2, 3)]


def test_failed_page_does_not_stop_other_pages() -> None:
    """한 페이지가 실패해도 나머지 페이지를 부르고, 실패한 페이지는 오류 코드로 구분된다."""
    error = PageExtractionError(PageExtractionErrorCode.TIMEOUT)
    extractor = FakeExtractor({2: error})

    outcomes = extract_pages(extractor, PAGES)

    assert extractor.requested_page_nos == [1, 2, 3]
    assert outcomes == [
        PageExtractionOutcome(page_no=1, result=RESULT),
        PageExtractionOutcome(page_no=2, error_code=PageExtractionErrorCode.TIMEOUT),
        PageExtractionOutcome(page_no=3, result=RESULT),
    ]


def test_unexpected_error_propagates() -> None:
    """PageExtractionError가 아닌 예외는 삼키지 않고 그대로 올린다."""
    error = RuntimeError("bug")

    with pytest.raises(RuntimeError) as raised:
        extract_pages(FakeExtractor({2: error}), PAGES)

    assert raised.value is error


def test_no_pages_returns_empty_list() -> None:
    """페이지가 없으면 포트를 부르지 않고 빈 목록을 돌려준다."""
    extractor = FakeExtractor()

    assert extract_pages(extractor, []) == []
    assert extractor.requested_page_nos == []
