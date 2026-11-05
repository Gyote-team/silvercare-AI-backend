"""문서 조각과 그 임베딩을 읽고 바꾸는 포트를 정의한다."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from app.domain.document.document_chunk import DocumentChunk, PageText


class ChunkIndexSkipReason(StrEnum):
    """색인을 건너뛴 이유."""

    DELETED = "DELETED"  # 문서가 없거나 삭제됨
    NOT_ANALYZED = "NOT_ANALYZED"  # 문서가 READY·NEEDS_REVIEW가 아니거나 최신 분석이 성공이 아님
    STALE_ANALYSIS = "STALE_ANALYSIS"  # 저장하려는 사이 더 새로운 분석이 생김
    NO_SOURCE_TEXT = "NO_SOURCE_TEXT"  # 원문이 있는 페이지가 없음
    NO_CHUNKS = "NO_CHUNKS"  # 원문에서 조각이 하나도 나오지 않아 기존 색인을 그대로 둠


@dataclass(frozen=True)
class ChunkIndexTarget:
    """조각으로 나눌 문서의 최신 성공 분석과 페이지 원문. pages는 page_no 순서다.

    pages는 원문이 있는 SUCCEEDED 페이지만 담고, failed_page_count는 FAILED 페이지 수다.
    """

    document_id: UUID
    analysis_id: UUID
    patient_id: UUID
    visit_id: UUID
    pages: tuple[PageText, ...]
    failed_page_count: int = 0


class DocumentChunkRepository(Protocol):
    """document_chunks와 DOCUMENT_CHUNK 임베딩을 다루는 인터페이스."""

    def find_index_target(self, document_id: UUID) -> ChunkIndexTarget | ChunkIndexSkipReason:
        """READY·NEEDS_REVIEW 문서의 최신 성공 분석과 원문이 있는 페이지를 읽는다.

        문서가 없거나 삭제됐으면 DELETED, 분석이 끝나지 않았거나 최신 분석이 SUCCEEDED가
        아니면 NOT_ANALYZED를 반환한다.
        """
        ...

    def replace_chunks(
        self,
        target: ChunkIndexTarget,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
        model_name: str,
    ) -> ChunkIndexSkipReason | None:
        """이 문서의 이전 조각·임베딩을 지우고 새 것을 한 트랜잭션에서 넣는다.

        저장하면 None이다. 그사이 문서가 삭제됐으면 DELETED, target 분석이 더 이상 최신 성공
        분석이 아니면 STALE_ANALYSIS를 반환하고 아무것도 바꾸지 않는다.
        중간에 실패하면 전부 롤백되어 이전 것이 남는다.
        """
        ...

    def find_unindexed_document_ids(self, *, limit: int) -> list[UUID]:
        """분석은 성공했지만 최신 분석의 조각 임베딩이 없는 문서 id를 limit건 읽는다."""
        ...
