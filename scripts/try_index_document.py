"""문서 하나를 실제 DB와 OpenAI 임베딩으로 한 번 색인해 보는 수동 확인 스크립트.

자동 테스트에 포함되지 않는다. 실행할 때마다 임베딩 비용이 들고 그 문서의 조각·임베딩이 바뀐다.
저장소 루트에서 실행한다. 분석이 READY나 NEEDS_REVIEW로 끝난 문서여야 한다.
조각 원문은 출력하지 않고 개수, 토큰 수 범위, 차원만 출력한다.

    python -m scripts.try_index_document <documentId>
"""

import sys
from uuid import UUID

from sqlalchemy import text

from app.application.use_cases.index_document_chunks import index_document_chunks
from app.core.config import get_settings
from app.infrastructure.db.repositories.document_chunk_repository import (
    SqlAlchemyDocumentChunkRepository,
)
from app.infrastructure.db.session import get_session_factory
from app.infrastructure.providers.embedding.openai_text_embedder import create_text_embedder

_SUMMARY = text(
    "SELECT count(*) AS chunks, count(e.id) AS embeddings,"
    " min(c.token_count) AS min_tokens, max(c.token_count) AS max_tokens,"
    " min(c.chunk_index) AS first_index, max(c.chunk_index) AS last_index,"
    " array_agg(DISTINCT vector_dims(e.embedding)) AS dims,"
    " array_agg(DISTINCT e.model_name) AS models"
    " FROM document_chunks c"
    " JOIN document_analyses a ON a.id = c.analysis_id"
    " LEFT JOIN embeddings e ON e.source_type = 'DOCUMENT_CHUNK' AND e.source_id = c.id"
    " WHERE a.document_id = :document_id"
)


def main() -> None:
    """인자로 받은 문서를 색인하고 저장 결과 요약을 출력한다."""
    if len(sys.argv) != 2:
        print("사용법: python -m scripts.try_index_document <documentId>")
        raise SystemExit(1)
    try:
        document_id = UUID(sys.argv[1])
    except ValueError:
        print("documentId는 UUID 형식이어야 합니다.")
        raise SystemExit(1) from None

    session_factory = get_session_factory()
    embedder = create_text_embedder(get_settings())
    print(f"document_id: {document_id}")
    try:
        outcome = index_document_chunks(
            document_id,
            repository=SqlAlchemyDocumentChunkRepository(session_factory),
            embedder=embedder,
        )
    except Exception as error:
        # 예외 메시지에는 원문이 들어 있을 수 있어 클래스 이름만 출력한다.
        print(f"예상 밖 오류: {type(error).__name__}")
        raise SystemExit(1) from None
    finally:
        embedder.close()

    print(f"status: {outcome.status}, chunks: {outcome.chunk_count}")
    if outcome.error_code is not None:
        print(f"error_code: {outcome.error_code}")
    with session_factory() as session:
        summary = session.execute(_SUMMARY, {"document_id": document_id}).one()
    print(
        f"stored chunks: {summary.chunks}, embeddings: {summary.embeddings},"
        f" chunk_index: {summary.first_index}..{summary.last_index},"
        f" tokens: {summary.min_tokens}..{summary.max_tokens},"
        f" dims: {summary.dims}, models: {summary.models}"
    )


if __name__ == "__main__":
    main()
