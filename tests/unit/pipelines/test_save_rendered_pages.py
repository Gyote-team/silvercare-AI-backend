"""이미지 저장 → document_pages 저장 단계 단위 테스트. 두 포트는 가짜로 대신한다."""

from collections.abc import Sequence
from datetime import datetime, timedelta
from uuid import UUID

import pytest

from app.application.ports.document_storage import RenderedPageWriteError
from app.domain.document.document_page import RenderedPage
from app.domain.document.page_split import PageImage
from app.pipelines.documents.steps.save_rendered_pages import save_rendered_pages

DOCUMENT_ID = UUID("11111111-2222-3333-4444-555555555555")
PAGES = [
    PageImage(page_no=1, png_bytes=b"page1", width_px=100, height_px=200),
    PageImage(page_no=2, png_bytes=b"page2", width_px=300, height_px=400),
]


class FakeWriter:
    """쓴 키와 바이트를 공용 호출 목록에 기록하는 가짜 포트. 정해 둔 키에서는 실패한다."""

    def __init__(self, calls: list[tuple], fail_on_key: str | None = None) -> None:
        self.calls = calls
        self.fail_on_key = fail_on_key

    def write_rendered_page(self, object_key: str, png_bytes: bytes) -> None:
        """호출을 기록한다. 실패하도록 정해 둔 키면 RenderedPageWriteError를 던진다."""
        if object_key == self.fail_on_key:
            raise RenderedPageWriteError(object_key)
        self.calls.append(("write", object_key, png_bytes))


class FakePageRepository:
    """저장 요청을 공용 호출 목록에 기록하는 가짜 포트."""

    def __init__(self, calls: list[tuple], error: Exception | None = None) -> None:
        self.calls = calls
        self.error = error

    def save_rendered_pages(
        self, document_id: UUID, pages: Sequence[RenderedPage], rendered_at: datetime
    ) -> None:
        """호출을 기록한다. 정해 둔 예외가 있으면 던진다."""
        if self.error is not None:
            raise self.error
        self.calls.append(("save", document_id, list(pages), rendered_at))


def test_writes_all_images_then_saves_pages_once() -> None:
    """페이지 순서대로 이미지를 모두 쓴 뒤 document_pages 저장을 한 번 부른다."""
    calls: list[tuple] = []

    save_rendered_pages(FakeWriter(calls), FakePageRepository(calls), DOCUMENT_ID, PAGES)

    key1 = "documents/11111111-2222-3333-4444-555555555555/pages/1.png"
    key2 = "documents/11111111-2222-3333-4444-555555555555/pages/2.png"
    assert calls[:2] == [("write", key1, b"page1"), ("write", key2, b"page2")]
    assert len(calls) == 3
    name, document_id, saved_pages, rendered_at = calls[2]
    assert (name, document_id) == ("save", DOCUMENT_ID)
    assert saved_pages == [
        RenderedPage(page_no=1, width_px=100, height_px=200, storage_key=key1),
        RenderedPage(page_no=2, width_px=300, height_px=400, storage_key=key2),
    ]
    assert rendered_at.utcoffset() == timedelta(0)


def test_image_write_error_propagates_and_skips_db() -> None:
    """이미지 쓰기가 실패하면 예외를 그대로 올리고 document_pages는 저장하지 않는다."""
    calls: list[tuple] = []
    writer = FakeWriter(calls, fail_on_key=f"documents/{DOCUMENT_ID}/pages/2.png")

    with pytest.raises(RenderedPageWriteError):
        save_rendered_pages(writer, FakePageRepository(calls), DOCUMENT_ID, PAGES)

    assert [call[0] for call in calls] == ["write"]


def test_page_save_error_propagates() -> None:
    """document_pages 저장 예외는 바꾸지 않고 그대로 올린다. 이미지는 이미 쓰여 있다."""
    calls: list[tuple] = []
    error = RuntimeError("db down")

    with pytest.raises(RuntimeError) as raised:
        save_rendered_pages(FakeWriter(calls), FakePageRepository(calls, error), DOCUMENT_ID, PAGES)

    assert raised.value is error
    assert [call[0] for call in calls] == ["write", "write"]
