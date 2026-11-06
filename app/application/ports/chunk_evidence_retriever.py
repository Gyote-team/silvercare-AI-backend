"""질문과 가까운 문서 조각을 권한 범위 안에서 검색하는 포트다."""

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True)
class RetrievedChunk:
    """AI 답변 근거로 사용할 검색된 문서 조각."""

    chunk_id: UUID
    document_id: UUID
    page_no: int | None
    content: str
    similarity: float


class ChunkEvidenceRetriever(Protocol):
    """환자와 허용 문서 범위를 강제하는 벡터 검색 인터페이스."""

    def retrieve(
        self,
        *,
        patient_id: UUID,
        allowed_document_ids: list[UUID],
        query_vector: list[float],
        model_name: str,
        top_k: int,
        min_similarity: float,
    ) -> list[RetrievedChunk]:
        """최신 성공 분석의 조각 중 질문과 가까운 조각을 반환한다."""
        ...
