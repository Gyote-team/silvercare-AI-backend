"""검색용으로 나눈 문서 조각과 그 입력인 페이지 원문을 정의한다."""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class PageText:
    """조각으로 나눌 페이지 한 장의 원문. page_no는 1부터 시작한다."""

    page_id: UUID
    page_no: int
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    """document_chunks 한 행에 담을 조각. chunk_index는 분석 안에서 0부터 연속이다."""

    chunk_index: int
    page_id: UUID
    content: str
    token_count: int
