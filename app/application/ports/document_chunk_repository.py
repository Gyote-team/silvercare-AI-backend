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
class StoredChunk:
    """이미 저장된 조각 하나. 임베딩만 다시 만들 때 쓴다."""

    chunk_id: UUID
    content: str


@dataclass(frozen=True)
class ChunkIndexTarget:
    """조각으로 나눌 문서의 최신 성공 분석과 페이지 원문. pages는 page_no 순서다.

    pages는 원문이 있는 SUCCEEDED 페이지만 담고, failed_page_count는 FAILED 페이지 수다.
    stored_chunk_count는 이 분석에 이미 저장된 조각 수, missing_chunks는 그중 현재 모델·차원의
    임베딩이 없는 조각(chunk_index 순)이다.
    """

    document_id: UUID
    analysis_id: UUID
    patient_id: UUID
    visit_id: UUID
    pages: tuple[PageText, ...]
    failed_page_count: int = 0
    stored_chunk_count: int = 0
    missing_chunks: tuple[StoredChunk, ...] = ()


class DocumentChunkRepository(Protocol):
    """document_chunks와 DOCUMENT_CHUNK 임베딩을 다루는 인터페이스.

    "색인됨"은 최신 분석의 조각이 하나 이상이고, 모든 조각에 model_name과 EMBEDDING_DIMENSIONS가
    같은 DOCUMENT_CHUNK 임베딩이 있는 상태다.
    """

    def find_index_target(
        self, document_id: UUID, model_name: str
    ) -> ChunkIndexTarget | ChunkIndexSkipReason:
        """READY·NEEDS_REVIEW 문서의 최신 성공 분석, 원문이 있는 페이지, 저장된 조각을 읽는다.

        문서가 없거나 삭제됐으면 DELETED, 분석이 끝나지 않았거나 최신 분석이 SUCCEEDED가
        아니면 NOT_ANALYZED를 반환한다.
        """
        ...

    def fill_missing_embeddings(
        self,
        target: ChunkIndexTarget,
        chunk_ids: Sequence[UUID],
        vectors: Sequence[Sequence[float]],
        model_name: str,
    ) -> ChunkIndexSkipReason | None:
        """target 분석의 조각 중 아직 현재 기준 임베딩이 없는 것에만 임베딩을 넣는다.

        조각은 지우지 않는다. 같은 조각의 다른 모델·차원 임베딩은 지우고 바꾼다.
        그사이 다른 실행이 채운 조각은 건너뛴다. 반환값은 replace_chunks와 같다.
        """
        ...

    def replace_chunks(
        self,
        target: ChunkIndexTarget,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
        model_name: str,
    ) -> ChunkIndexSkipReason | None:
        """이 문서의 조각 임베딩을 지우고 새 조각·임베딩을 한 트랜잭션에서 넣는다.

        이전 분석의 조각 행은 인용될 수 있어 남기고 임베딩만 지운다.
        저장하면 None이다. 그사이 문서가 삭제됐으면 DELETED, target 분석이 더 이상 최신 성공
        분석이 아니면 STALE_ANALYSIS를 반환하고 아무것도 바꾸지 않는다.
        중간에 실패하면 전부 롤백되어 이전 것이 남는다.
        """
        ...

    def find_unindexed_document_ids(self, *, model_name: str, limit: int) -> list[UUID]:
        """분석은 성공했지만 최신 분석이 model_name 기준으로 색인되지 않은 문서 id를 읽는다."""
        ...
