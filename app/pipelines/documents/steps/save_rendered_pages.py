"""페이지 이미지를 저장소에 쓰고 document_pages에 기록하는 문서 파이프라인 단계."""

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

from app.application.ports.document_page_repository import DocumentPageRepository
from app.application.ports.document_storage import RenderedPageWriter
from app.domain.document.document_page import RenderedPage, rendered_page_key
from app.domain.document.page_split import PageImage


def save_rendered_pages(
    writer: RenderedPageWriter,
    page_repository: DocumentPageRepository,
    document_id: UUID,
    pages: Sequence[PageImage],
) -> None:
    """모든 페이지 이미지를 먼저 쓰고, 그다음 document_pages에 한 번에 저장한다."""
    rendered_pages: list[RenderedPage] = []
    # DB가 없는 파일을 가리키지 않게 이미지를 모두 쓴 뒤에 DB에 저장한다.
    for page in pages:
        storage_key = rendered_page_key(document_id, page.page_no)
        writer.write_rendered_page(storage_key, page.png_bytes)
        rendered_pages.append(
            RenderedPage(
                page_no=page.page_no,
                width_px=page.width_px,
                height_px=page.height_px,
                storage_key=storage_key,
            )
        )
    page_repository.save_rendered_pages(document_id, rendered_pages, datetime.now(UTC))
