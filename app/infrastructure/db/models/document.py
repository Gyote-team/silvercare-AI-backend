"""Spring Flyway가 만든 documents 테이블 매핑. 분석이 읽고 쓰는 컬럼만 적는다."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import String, Uuid
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.models.base import Base


class DocumentModel(Base):
    """업로드된 문서 한 건. 행은 Spring이 만들고, 여기서는 상태 컬럼만 바꾼다."""

    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    storage_key: Mapped[str] = mapped_column(String(500))
    mime_type: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20))
    status_changed_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True, precision=6))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True, precision=6))
    deleted_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True, precision=6))
