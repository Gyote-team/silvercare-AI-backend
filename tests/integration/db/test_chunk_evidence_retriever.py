"""채팅 청크 검색이 환자·문서·최신 분석 범위를 지키는지 검증한다."""

from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from app.infrastructure.db.repositories.chunk_evidence_retriever import (
    SqlAlchemyChunkEvidenceRetriever,
)
from app.infrastructure.db.repositories.document_chunk_repository import (
    SqlAlchemyDocumentChunkRepository,
)
from tests.integration.db.test_document_chunk_repository import (
    DIMENSIONS,
    MODEL_NAME,
    add_newer_analysis,
    make_ready_document,
    require_target,
    store,
)


def test_retrieval_returns_only_allowed_document_chunks(
    db_session: Session, session_factory: sessionmaker[Session]
) -> None:
    chunk_repository = SqlAlchemyDocumentChunkRepository(session_factory)
    allowed = make_ready_document(db_session, {1: "허용된 검사 결과"})
    denied = make_ready_document(db_session, {1: "다른 문서 결과"})
    store(chunk_repository, require_target(chunk_repository, allowed), "허용된 검사 결과")
    store(chunk_repository, require_target(chunk_repository, denied), "다른 문서 결과")
    patient_id = db_session.execute(
        text("SELECT patient_id FROM documents WHERE id = :id"), {"id": allowed}
    ).scalar_one()

    results = SqlAlchemyChunkEvidenceRetriever(session_factory).retrieve(
        patient_id=patient_id,
        allowed_document_ids=[allowed],
        query_vector=[0.1] * DIMENSIONS,
        model_name=MODEL_NAME,
        top_k=8,
        min_similarity=0.30,
    )

    assert [(result.document_id, result.content) for result in results] == [
        (allowed, "허용된 검사 결과")
    ]


def test_retrieval_excludes_chunks_from_superseded_analysis(
    db_session: Session, session_factory: sessionmaker[Session]
) -> None:
    chunk_repository = SqlAlchemyDocumentChunkRepository(session_factory)
    document_id = make_ready_document(db_session, {1: "이전 분석 원문"})
    store(chunk_repository, require_target(chunk_repository, document_id), "이전 분석 조각")
    add_newer_analysis(db_session, document_id, "SUCCEEDED")
    patient_id = db_session.execute(
        text("SELECT patient_id FROM documents WHERE id = :id"), {"id": document_id}
    ).scalar_one()

    results = SqlAlchemyChunkEvidenceRetriever(session_factory).retrieve(
        patient_id=patient_id,
        allowed_document_ids=[document_id],
        query_vector=[0.1] * DIMENSIONS,
        model_name=MODEL_NAME,
        top_k=8,
        min_similarity=0.30,
    )

    assert results == []
