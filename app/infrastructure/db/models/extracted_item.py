"""Spring Flyway가 만든 extracted_items 테이블 매핑. V2를 따른다."""

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import ForeignKey, Numeric, String, Text, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.models.base import Base


class ExtractedItemModel(Base):
    """문서에서 추출한 항목 하나. 어느 분석의 어느 페이지에서 나왔는지 가리킨다."""

    __tablename__ = "extracted_items"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    analysis_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("document_analyses.id", name="extracted_items_analysis_fk")
    )
    page_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("document_pages.id", name="extracted_items_page_fk")
    )
    item_type: Mapped[str] = mapped_column(String(30))
    normalized_name: Mapped[str | None] = mapped_column(String(200))
    raw_value: Mapped[str | None] = mapped_column(String(500))
    normalized_value: Mapped[str | None] = mapped_column(String(500))
    unit: Mapped[str | None] = mapped_column(String(50))
    reference_range: Mapped[str | None] = mapped_column(String(100))
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    source_text: Mapped[str] = mapped_column(Text)
    # 박스가 없으면 JSON null이 아니라 SQL NULL로 넣는다.
    source_box: Mapped[dict[str, Any] | None] = mapped_column(JSONB(none_as_null=True))
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True, precision=6), server_default=func.current_timestamp()
    )
