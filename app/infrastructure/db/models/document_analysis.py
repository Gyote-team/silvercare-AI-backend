"""Spring Flyway가 만든 document_analyses 테이블 매핑. 분석이 읽고 쓰는 컬럼만 적는다."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Boolean, ForeignKey, String, Text, Uuid
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.models.base import Base


class DocumentAnalysisModel(Base):
    """문서 분석 작업 한 건. 행은 Spring이 PENDING으로 만든다."""

    __tablename__ = "document_analyses"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    document_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("documents.id", name="document_analyses_document_fk")
    )
    model_version: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20))
    started_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True, precision=6))
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True, precision=6))
    error_message: Mapped[str | None] = mapped_column(Text)
    retryable: Mapped[bool] = mapped_column(Boolean)
    error_code: Mapped[str | None] = mapped_column(String(100))
    failed_step: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True, precision=6))
