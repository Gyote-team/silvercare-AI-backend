"""document_pages 한 행에 담는 페이지 규칙: OCR 상태, 렌더링 이미지 키와 형식."""

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

RENDERED_PAGE_MIME_TYPE = "image/png"


class PageOcrStatus(StrEnum):
    """페이지 OCR 상태. document_pages.ocr_status의 CHECK 제약 값과 같다."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class RenderedPage:
    """렌더링 이미지를 저장한 페이지 한 장의 크기와 저장 키. page_no는 1부터 시작한다."""

    page_no: int
    width_px: int
    height_px: int
    storage_key: str


def rendered_page_key(document_id: UUID, page_no: int) -> str:
    """렌더링 이미지 키를 만든다.

    Spring 원본 키 documents/{documentId}/original.{ext}와 폴더가 같다.
    """
    return f"documents/{document_id}/pages/{page_no}.png"
