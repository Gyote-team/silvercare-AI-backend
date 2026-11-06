"""PostgreSQL pgvector를 사용하는 채팅 문서 조각 검색 구현."""

from uuid import UUID

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session, sessionmaker

from app.application.ports.chunk_evidence_retriever import RetrievedChunk
from app.application.ports.text_embedder import EMBEDDING_DIMENSIONS

_SEARCH = text(
    """
    SELECT c.id AS chunk_id,
           d.id AS document_id,
           p.page_no AS page_no,
           c.content AS content,
           1 - (e.embedding <=> CAST(:query_vector AS vector)) AS similarity
      FROM embeddings e
      JOIN document_chunks c ON c.id = e.source_id
      JOIN document_analyses a ON a.id = c.analysis_id
      JOIN documents d ON d.id = a.document_id
      LEFT JOIN document_pages p ON p.id = c.page_id
     WHERE e.source_type = 'DOCUMENT_CHUNK'
       AND e.patient_id = :patient_id
       AND e.model_name = :model_name
       AND e.dimensions = :dimensions
       AND d.id IN :document_ids
       AND d.patient_id = :patient_id
       AND d.deleted_at IS NULL
       AND d.status IN ('READY', 'NEEDS_REVIEW')
       AND a.status = 'SUCCEEDED'
       AND a.id = (
            SELECT latest.id
              FROM document_analyses latest
             WHERE latest.document_id = d.id
               AND latest.status = 'SUCCEEDED'
             ORDER BY latest.created_at DESC, latest.id DESC
             LIMIT 1
       )
       AND 1 - (e.embedding <=> CAST(:query_vector AS vector)) >= :min_similarity
     ORDER BY e.embedding <=> CAST(:query_vector AS vector), c.id
     LIMIT :top_k
    """
).bindparams(bindparam("document_ids", expanding=True))


class SqlAlchemyChunkEvidenceRetriever:
    """세션 하나에서 읽기 전용 유사도 검색을 수행한다."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

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
        if not allowed_document_ids:
            return []
        if len(query_vector) != EMBEDDING_DIMENSIONS:
            raise ValueError("질문 임베딩 차원이 올바르지 않습니다")
        vector = "[" + ",".join(str(float(value)) for value in query_vector) + "]"
        with self._session_factory() as session:
            rows = session.execute(
                _SEARCH,
                {
                    "patient_id": patient_id,
                    "document_ids": list(dict.fromkeys(allowed_document_ids)),
                    "query_vector": vector,
                    "model_name": model_name,
                    "dimensions": EMBEDDING_DIMENSIONS,
                    "min_similarity": min_similarity,
                    "top_k": top_k,
                },
            ).all()
        allowed = set(allowed_document_ids)
        results = [
            RetrievedChunk(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                page_no=row.page_no,
                content=row.content,
                similarity=float(row.similarity),
            )
            for row in rows
        ]
        if any(result.document_id not in allowed for result in results):
            raise PermissionError("허용 범위 밖 문서 조각이 검색되었습니다")
        return results
