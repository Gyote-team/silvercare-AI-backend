"""document_pages 저장소 테스트. 실제 PostgreSQL에 쓰고 테스트가 끝나면 롤백한다."""

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import Row, select, text
from sqlalchemy.orm import Session

from app.domain.document.document_page import (
    RENDERED_PAGE_MIME_TYPE,
    PageOcrStatus,
    RenderedPage,
    rendered_page_key,
)
from app.infrastructure.db.models.document_page import DocumentPageModel
from app.infrastructure.db.repositories.document_page_repository import (
    SqlAlchemyDocumentPageRepository,
)

FIRST_RENDERED_AT = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
SECOND_RENDERED_AT = datetime(2026, 10, 2, 9, 0, tzinfo=UTC)


def make_pages(document_id: UUID, width_px: int) -> list[RenderedPage]:
    """세로가 가로의 두 배인 2페이지 목록을 만든다."""
    return [
        RenderedPage(
            page_no=page_no,
            width_px=width_px,
            height_px=width_px * 2,
            storage_key=rendered_page_key(document_id, page_no),
        )
        for page_no in (1, 2)
    ]


def fetch_rows(session: Session, document_id: UUID) -> Sequence[Row]:
    """문서의 document_pages 행을 page_no 순서로 읽는다."""
    table = DocumentPageModel.__table__
    return session.execute(
        select(table).where(table.c.document_id == document_id).order_by(table.c.page_no)
    ).all()


def test_first_save_inserts_one_row_per_page(db_session: Session, document_id: UUID) -> None:
    """처음 저장하면 페이지 수만큼 행이 생기고 rendered_* 세 컬럼이 모두 채워진다."""
    repository = SqlAlchemyDocumentPageRepository(db_session)

    repository.save_rendered_pages(document_id, make_pages(document_id, 100), FIRST_RENDERED_AT)

    rows = fetch_rows(db_session, document_id)
    assert [row.page_no for row in rows] == [1, 2]
    for row in rows:
        assert (row.width_px, row.height_px) == (100, 200)
        assert row.rendered_storage_key == rendered_page_key(document_id, row.page_no)
        assert row.rendered_mime_type == RENDERED_PAGE_MIME_TYPE
        assert row.rendered_at == FIRST_RENDERED_AT
        assert row.ocr_status == PageOcrStatus.PENDING
        assert row.ocr_text is None


def test_reprocess_updates_rendered_columns_only(db_session: Session, document_id: UUID) -> None:
    """다시 저장하면 행 수와 id는 그대로고, 크기와 rendered_at만 바뀐다. OCR 값은 남는다."""
    repository = SqlAlchemyDocumentPageRepository(db_session)
    repository.save_rendered_pages(document_id, make_pages(document_id, 100), FIRST_RENDERED_AT)
    db_session.execute(
        text(
            "UPDATE document_pages SET ocr_status = :status, ocr_text = 'text'"
            " WHERE document_id = :document_id"
        ),
        {"status": PageOcrStatus.SUCCEEDED, "document_id": document_id},
    )
    before = fetch_rows(db_session, document_id)

    repository.save_rendered_pages(document_id, make_pages(document_id, 300), SECOND_RENDERED_AT)

    after = fetch_rows(db_session, document_id)
    assert [row.id for row in after] == [row.id for row in before]
    assert [row.created_at for row in after] == [row.created_at for row in before]
    for row in after:
        assert (row.width_px, row.height_px) == (300, 600)
        assert row.rendered_at == SECOND_RENDERED_AT
        assert row.rendered_storage_key == rendered_page_key(document_id, row.page_no)
        assert row.rendered_mime_type == RENDERED_PAGE_MIME_TYPE
        assert row.ocr_status == PageOcrStatus.SUCCEEDED
        assert row.ocr_text == "text"


def test_empty_pages_inserts_nothing(db_session: Session, document_id: UUID) -> None:
    """페이지 목록이 비어 있으면 아무 행도 만들지 않는다."""
    SqlAlchemyDocumentPageRepository(db_session).save_rendered_pages(
        document_id, [], FIRST_RENDERED_AT
    )

    assert fetch_rows(db_session, document_id) == []
