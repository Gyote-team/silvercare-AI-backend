"""읽기 → 페이지 분리 단계 단위 테스트. 저장소는 가짜 포트로 대신한다."""

import pymupdf
import pytest

from app.application.ports.document_storage import DocumentNotFoundError
from app.domain.document.page_split import PageSplitError, PageSplitErrorCode
from app.pipelines.documents.steps.load_document_pages import load_document_pages

OBJECT_KEY = "documents/1/original.pdf"
PDF = "application/pdf"


class FakeReader:
    """받은 objectKey를 기록하고 정해 둔 바이트를 돌려주는 가짜 포트."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.requested_keys: list[str] = []

    def read_original(self, object_key: str) -> bytes:
        """요청받은 키를 기록하고 바이트를 반환한다."""
        self.requested_keys.append(object_key)
        return self.data


class MissingReader:
    """어떤 키든 파일이 없다고 답하는 가짜 포트."""

    def read_original(self, object_key: str) -> bytes:
        """항상 DocumentNotFoundError를 던진다."""
        raise DocumentNotFoundError(object_key)


def make_two_page_pdf() -> bytes:
    """가로 100, 세로 200 포인트 페이지 2장짜리 PDF를 만든다."""
    document = pymupdf.open()
    document.new_page(width=100, height=200)
    document.new_page(width=100, height=200)
    return document.tobytes()


def test_reads_by_object_key_then_splits_pages() -> None:
    """포트에 objectKey를 그대로 넘기고, 읽은 바이트를 설정값대로 페이지로 나눈다."""
    reader = FakeReader(make_two_page_pdf())

    pages = load_document_pages(
        reader, OBJECT_KEY, PDF, render_dpi=144, max_pages=2, max_side_px=2048
    )

    assert reader.requested_keys == [OBJECT_KEY]
    assert [page.page_no for page in pages] == [1, 2]
    assert (pages[0].width_px, pages[0].height_px) == (200, 400)


def test_read_error_propagates() -> None:
    """포트의 읽기 예외는 바꾸지 않고 그대로 올린다."""
    with pytest.raises(DocumentNotFoundError):
        load_document_pages(
            MissingReader(), OBJECT_KEY, PDF, render_dpi=72, max_pages=2, max_side_px=2048
        )


def test_split_error_propagates() -> None:
    """분리 오류는 오류 코드와 함께 그대로 올린다."""
    reader = FakeReader(b"not a pdf")

    with pytest.raises(PageSplitError) as error:
        load_document_pages(reader, OBJECT_KEY, PDF, render_dpi=72, max_pages=2, max_side_px=2048)

    assert error.value.code is PageSplitErrorCode.CORRUPTED_FILE


def test_page_limit_is_passed_to_splitter() -> None:
    """max_pages가 분리기에 전달돼 초과 문서를 거절한다."""
    reader = FakeReader(make_two_page_pdf())

    with pytest.raises(PageSplitError) as error:
        load_document_pages(reader, OBJECT_KEY, PDF, render_dpi=72, max_pages=1, max_side_px=2048)

    assert error.value.code is PageSplitErrorCode.PAGE_LIMIT_EXCEEDED


def test_side_limit_is_passed_to_splitter() -> None:
    """max_side_px가 분리기에 전달돼 긴 변이 그 크기로 줄어든다."""
    reader = FakeReader(make_two_page_pdf())

    pages = load_document_pages(
        reader, OBJECT_KEY, PDF, render_dpi=144, max_pages=2, max_side_px=100
    )

    assert (pages[0].width_px, pages[0].height_px) == (50, 100)
