"""document_pages에 페이지 렌더링 결과를 저장하는 포트를 정의한다."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol
from uuid import UUID

from app.domain.document.document_page import RenderedPage


class DocumentPageRepository(Protocol):
    """문서의 페이지 행을 저장하는 인터페이스."""

    def save_rendered_pages(
        self, document_id: UUID, pages: Sequence[RenderedPage], rendered_at: datetime
    ) -> None:
        """페이지마다 한 행을 저장한다. 같은 문서를 다시 저장해도 행이 늘지 않는다."""
        ...
