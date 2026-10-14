"""document_pages에 페이지 렌더링 결과를 저장하는 SQLAlchemy 구현."""

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.domain.document.document_page import RENDERED_PAGE_MIME_TYPE, RenderedPage
from app.infrastructure.db.models.document_page import (
    DOCUMENT_PAGE_KEY_CONSTRAINT,
    DocumentPageModel,
)

# 다시 처리할 때 바꾸는 컬럼. id, ocr_status, ocr_text, created_at은 그대로 둔다.
_UPDATED_ON_REPROCESS = (
    "width_px",
    "height_px",
    "rendered_storage_key",
    "rendered_mime_type",
    "rendered_at",
)


class SqlAlchemyDocumentPageRepository:
    """받은 session으로 document_pages를 쓴다. commit은 session을 연 쪽이 한다."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_rendered_pages(
        self, document_id: UUID, pages: Sequence[RenderedPage], rendered_at: datetime
    ) -> None:
        """페이지마다 한 행을 넣고, 이미 있는 페이지는 크기와 렌더링 컬럼만 갱신한다."""
        if not pages:
            return
        statement = insert(DocumentPageModel).values(
            [
                {
                    "id": uuid4(),
                    "document_id": document_id,
                    "page_no": page.page_no,
                    "width_px": page.width_px,
                    "height_px": page.height_px,
                    "rendered_storage_key": page.storage_key,
                    "rendered_mime_type": RENDERED_PAGE_MIME_TYPE,
                    "rendered_at": rendered_at,
                }
                for page in pages
            ]
        )
        self._session.execute(
            statement.on_conflict_do_update(
                constraint=DOCUMENT_PAGE_KEY_CONSTRAINT,
                set_={name: statement.excluded[name] for name in _UPDATED_ON_REPROCESS},
            )
        )
