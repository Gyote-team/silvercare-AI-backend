"""Spring Flyway가 만든 document_pages 테이블 매핑. V2와 V202610061000을 따른다."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.document.document_page import PageOcrStatus
from app.infrastructure.db.models.base import Base

DOCUMENT_PAGE_KEY_CONSTRAINT = "document_pages_document_page_key"

_OCR_STATUS_VALUES = ", ".join(f"'{status}'" for status in PageOcrStatus)


class DocumentPageModel(Base):
    """문서의 페이지 한 장. (document_id, page_no)가 유일하다."""

    __tablename__ = "document_pages"
    __table_args__ = (
        UniqueConstraint("document_id", "page_no", name=DOCUMENT_PAGE_KEY_CONSTRAINT),
        CheckConstraint(
            f"ocr_status IN ({_OCR_STATUS_VALUES})", name="document_pages_status_check"
        ),
        CheckConstraint(
            "(rendered_storage_key IS NULL AND rendered_mime_type IS NULL"
            " AND rendered_at IS NULL)"
            " OR (rendered_storage_key IS NOT NULL AND rendered_mime_type IS NOT NULL"
            " AND rendered_at IS NOT NULL)",
            name="document_pages_rendered_asset_check",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    # documents 테이블은 매핑하지 않으므로 타입을 직접 적는다.
    document_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("documents.id", name="document_pages_document_fk")
    )
    page_no: Mapped[int] = mapped_column(Integer)
    ocr_status: Mapped[str] = mapped_column(
        String(20), server_default=text(f"'{PageOcrStatus.PENDING}'")
    )
    ocr_text: Mapped[str | None] = mapped_column(Text)
    width_px: Mapped[int | None] = mapped_column(Integer)
    height_px: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True, precision=6), server_default=func.current_timestamp()
    )
    rendered_storage_key: Mapped[str | None] = mapped_column(String(500))
    rendered_mime_type: Mapped[str | None] = mapped_column(String(100))
    rendered_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True, precision=6))
