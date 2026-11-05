"""문서 조각·임베딩 저장소 테스트. 실제 PostgreSQL에 쓰고 테스트가 끝나면 롤백한다."""

from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Row, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.application.ports.document_chunk_repository import ChunkIndexSkipReason, ChunkIndexTarget
from app.application.use_cases.index_document_chunks import (
    ChunkIndexStatus,
    index_document_chunks,
)
from app.domain.document.document_chunk import DocumentChunk, PageText
from app.infrastructure.db.repositories.document_chunk_repository import (
    SqlAlchemyDocumentChunkRepository,
)
from tests.integration.db.conftest import insert_analysis, insert_document

DIMENSIONS = 768
MODEL_NAME = "text-embedding-3-small"


def make_ready_document(db_session: Session, page_texts: dict[int, str | None]) -> UUID:
    """분석이 성공해 READY가 된 문서와 페이지를 만든다. 원문이 None이면 SKIPPED 페이지다."""
    document_id = insert_document(db_session)
    analysis_id = insert_analysis(db_session, document_id)
    db_session.execute(
        text("UPDATE documents SET status = 'READY' WHERE id = :id"), {"id": document_id}
    )
    db_session.execute(
        text("UPDATE document_analyses SET status = 'SUCCEEDED' WHERE id = :id"),
        {"id": analysis_id},
    )
    for page_no, ocr_text in page_texts.items():
        db_session.execute(
            text(
                "INSERT INTO document_pages (id, document_id, page_no, ocr_status, ocr_text)"
                " VALUES (:id, :document_id, :page_no, :status, :ocr_text)"
            ),
            {
                "id": uuid4(),
                "document_id": document_id,
                "page_no": page_no,
                "status": "SUCCEEDED" if ocr_text is not None else "SKIPPED",
                "ocr_text": ocr_text,
            },
        )
    return document_id


def add_newer_analysis(db_session: Session, document_id: UUID, status: str) -> UUID:
    """같은 문서에 더 늦게 만든 분석 행을 넣는다. 한 트랜잭션 안이라 created_at을 직접 늦춘다."""
    analysis_id = insert_analysis(db_session, document_id)
    db_session.execute(
        text(
            "UPDATE document_analyses"
            " SET status = :status, created_at = created_at + INTERVAL '1 minute'"
            " WHERE id = :id"
        ),
        {"id": analysis_id, "status": status},
    )
    return analysis_id


def make_chunks(target: ChunkIndexTarget, *contents: str) -> list[DocumentChunk]:
    """첫 페이지에 붙은 합성 조각을 만든다."""
    return [
        DocumentChunk(
            chunk_index=index, page_id=target.pages[0].page_id, content=content, token_count=3
        )
        for index, content in enumerate(contents)
    ]


def vectors(count: int) -> list[list[float]]:
    """조각마다 다른 768차원 합성 벡터를 만든다."""
    return [[float(index + 1) / 10] * DIMENSIONS for index in range(count)]


def stored_rows(db_session: Session, document_id: UUID) -> list[Row]:
    """문서의 조각과 그 임베딩을 조각 번호 순서로 읽는다."""
    return db_session.execute(
        text(
            "SELECT c.analysis_id, c.chunk_index, c.content, c.token_count, c.page_id,"
            " e.patient_id, e.visit_id, e.model_name, e.dimensions,"
            " vector_dims(e.embedding) AS dims"
            " FROM document_chunks c"
            " JOIN document_analyses a ON a.id = c.analysis_id"
            " LEFT JOIN embeddings e ON e.source_type = 'DOCUMENT_CHUNK' AND e.source_id = c.id"
            " WHERE a.document_id = :id ORDER BY c.chunk_index"
        ),
        {"id": document_id},
    ).all()


def store(
    repository: SqlAlchemyDocumentChunkRepository, target: ChunkIndexTarget, *contents: str
) -> ChunkIndexSkipReason | None:
    """합성 조각과 벡터로 replace_chunks를 부른다. 저장하면 None이다."""
    return repository.replace_chunks(
        target, make_chunks(target, *contents), vectors(len(contents)), MODEL_NAME
    )


@pytest.fixture
def repository(session_factory: sessionmaker[Session]) -> SqlAlchemyDocumentChunkRepository:
    """테스트 트랜잭션에 묶인 저장소."""
    return SqlAlchemyDocumentChunkRepository(session_factory)


def require_target(
    repository: SqlAlchemyDocumentChunkRepository, document_id: UUID
) -> ChunkIndexTarget:
    """색인 대상을 읽고, 없으면 테스트를 실패시킨다."""
    target = repository.find_index_target(document_id, MODEL_NAME)
    assert isinstance(target, ChunkIndexTarget)
    return target


def test_index_target_has_ids_and_text_pages_in_order(
    db_session: Session, repository: SqlAlchemyDocumentChunkRepository
) -> None:
    """대상에는 최신 성공 분석, 환자·방문 id, 원문이 있는 페이지만 page_no 순서로 담긴다."""
    document_id = make_ready_document(db_session, {2: "둘째 쪽", 1: "첫째 쪽", 3: None})
    expected = db_session.execute(
        text(
            "SELECT d.patient_id, d.visit_id, a.id AS analysis_id FROM documents d"
            " JOIN document_analyses a ON a.document_id = d.id WHERE d.id = :id"
        ),
        {"id": document_id},
    ).one()

    target = require_target(repository, document_id)

    assert (target.patient_id, target.visit_id, target.analysis_id) == tuple(expected)
    assert [(page.page_no, page.text) for page in target.pages] == [(1, "첫째 쪽"), (2, "둘째 쪽")]
    assert all(isinstance(page, PageText) for page in target.pages)


def test_no_target_when_latest_analysis_is_not_succeeded(
    db_session: Session, repository: SqlAlchemyDocumentChunkRepository
) -> None:
    """최신 분석이 SUCCEEDED가 아니면 NOT_ANALYZED다."""
    document_id = make_ready_document(db_session, {1: "원문"})
    add_newer_analysis(db_session, document_id, "RUNNING")

    assert (
        repository.find_index_target(document_id, MODEL_NAME) == ChunkIndexSkipReason.NOT_ANALYZED
    )


def test_no_target_when_document_is_not_ready(
    db_session: Session, repository: SqlAlchemyDocumentChunkRepository
) -> None:
    """분석 중인 문서는 NOT_ANALYZED다."""
    document_id = make_ready_document(db_session, {1: "원문"})
    db_session.execute(
        text("UPDATE documents SET status = 'PROCESSING' WHERE id = :id"), {"id": document_id}
    )

    assert (
        repository.find_index_target(document_id, MODEL_NAME) == ChunkIndexSkipReason.NOT_ANALYZED
    )


def test_no_target_when_document_is_missing(repository: SqlAlchemyDocumentChunkRepository) -> None:
    """없는 문서는 삭제된 문서와 같이 DELETED다."""
    assert repository.find_index_target(uuid4(), MODEL_NAME) == ChunkIndexSkipReason.DELETED


def test_index_target_counts_failed_pages(
    db_session: Session, repository: SqlAlchemyDocumentChunkRepository
) -> None:
    """FAILED 페이지 수를 함께 읽는다. SKIPPED 페이지는 세지 않는다."""
    document_id = make_ready_document(db_session, {1: "원문", 2: None, 3: "실패할 쪽"})
    db_session.execute(
        text(
            "UPDATE document_pages SET ocr_status = 'FAILED', ocr_text = NULL"
            " WHERE document_id = :id AND page_no = 3"
        ),
        {"id": document_id},
    )

    target = require_target(repository, document_id)

    assert (len(target.pages), target.failed_page_count) == (1, 1)


# 문서 삭제 방식 두 가지. Spring은 status를 DELETED로 바꾸거나 deleted_at을 채운다.
DELETIONS = {
    "deleted_at": "UPDATE documents SET deleted_at = CURRENT_TIMESTAMP WHERE id = :id",
    "status": "UPDATE documents SET status = 'DELETED' WHERE id = :id",
}


def delete_document(db_session: Session, document_id: UUID, kind: str) -> None:
    """주어진 방식으로 문서를 삭제 상태로 만든다."""
    db_session.execute(text(DELETIONS[kind]), {"id": document_id})


@pytest.mark.parametrize("kind", DELETIONS)
def test_no_target_when_document_is_deleted(
    db_session: Session, repository: SqlAlchemyDocumentChunkRepository, kind: str
) -> None:
    """deleted_at이 있거나 status가 DELETED인 문서는 DELETED다."""
    document_id = make_ready_document(db_session, {1: "원문"})
    delete_document(db_session, document_id, kind)

    assert repository.find_index_target(document_id, MODEL_NAME) == ChunkIndexSkipReason.DELETED


def test_replace_stores_chunks_with_embeddings(
    db_session: Session, repository: SqlAlchemyDocumentChunkRepository
) -> None:
    """조각과 768차원 임베딩이 환자·방문·모델 이름과 함께 저장된다."""
    document_id = make_ready_document(db_session, {1: "원문"})
    target = require_target(repository, document_id)

    assert store(repository, target, "첫 조각", "둘째 조각") is None

    rows = stored_rows(db_session, document_id)
    assert [(row.chunk_index, row.content) for row in rows] == [(0, "첫 조각"), (1, "둘째 조각")]
    for row in rows:
        assert row.analysis_id == target.analysis_id
        assert row.page_id == target.pages[0].page_id
        assert (row.patient_id, row.visit_id) == (target.patient_id, target.visit_id)
        assert (row.model_name, row.dimensions, row.dims) == (MODEL_NAME, DIMENSIONS, DIMENSIONS)


def test_rerun_of_same_analysis_replaces_previous_chunks(
    db_session: Session, repository: SqlAlchemyDocumentChunkRepository
) -> None:
    """같은 분석으로 다시 저장하면 유일 키에 걸리지 않고 이전 조각·임베딩이 바뀐다."""
    document_id = make_ready_document(db_session, {1: "원문"})
    target = require_target(repository, document_id)
    store(repository, target, "이전 1", "이전 2", "이전 3")

    assert store(repository, target, "새 조각") is None

    assert [row.content for row in stored_rows(db_session, document_id)] == ["새 조각"]
    assert count_chunk_embeddings(db_session, document_id) == 1


def test_new_analysis_removes_chunks_of_previous_analysis(
    db_session: Session, repository: SqlAlchemyDocumentChunkRepository
) -> None:
    """재분석으로 새 분석 행이 생기면 이전 분석의 조각과 임베딩이 지워진다."""
    document_id = make_ready_document(db_session, {1: "원문"})
    store(repository, require_target(repository, document_id), "이전 조각")
    new_analysis_id = add_newer_analysis(db_session, document_id, "SUCCEEDED")

    assert store(repository, require_target(repository, document_id), "새 조각") is None

    rows = stored_rows(db_session, document_id)
    assert [(row.analysis_id, row.content) for row in rows] == [(new_analysis_id, "새 조각")]
    assert count_chunk_embeddings(db_session, document_id) == 1


def test_failed_insert_keeps_previous_chunks(
    db_session: Session, repository: SqlAlchemyDocumentChunkRepository
) -> None:
    """새 조각을 넣다 실패하면 전부 롤백되어 이전 조각과 임베딩이 남는다."""
    document_id = make_ready_document(db_session, {1: "원문"})
    target = require_target(repository, document_id)
    store(repository, target, "이전 조각")
    # 없는 페이지를 가리켜 FK에 걸리게 한다.
    broken = [DocumentChunk(chunk_index=0, page_id=uuid4(), content="새 조각", token_count=1)]

    with pytest.raises(IntegrityError):
        repository.replace_chunks(target, broken, vectors(1), MODEL_NAME)

    assert [row.content for row in stored_rows(db_session, document_id)] == ["이전 조각"]
    assert count_chunk_embeddings(db_session, document_id) == 1


def test_cited_chunk_blocks_replace_and_keeps_previous(
    db_session: Session, repository: SqlAlchemyDocumentChunkRepository
) -> None:
    """이전 조각이 인용돼 지울 수 없으면 전부 롤백되어 이전 것이 그대로 남는다."""
    document_id = make_ready_document(db_session, {1: "원문"})
    target = require_target(repository, document_id)
    store(repository, target, "인용된 조각")
    cite_first_chunk(db_session, document_id)

    with pytest.raises(IntegrityError):
        store(repository, target, "새 조각")

    assert [row.content for row in stored_rows(db_session, document_id)] == ["인용된 조각"]
    assert count_chunk_embeddings(db_session, document_id) == 1


def test_superseded_analysis_changes_nothing(
    db_session: Session, repository: SqlAlchemyDocumentChunkRepository
) -> None:
    """그사이 새 분석이 생겼으면 이전 대상으로 저장하지 않고 STALE_ANALYSIS를 반환한다."""
    document_id = make_ready_document(db_session, {1: "원문"})
    old_target = require_target(repository, document_id)
    add_newer_analysis(db_session, document_id, "SUCCEEDED")

    assert store(repository, old_target, "늦은 조각") == ChunkIndexSkipReason.STALE_ANALYSIS
    assert stored_rows(db_session, document_id) == []


@pytest.mark.parametrize("kind", DELETIONS)
def test_deleted_document_changes_nothing(
    db_session: Session, repository: SqlAlchemyDocumentChunkRepository, kind: str
) -> None:
    """그사이 deleted_at이 채워졌거나 DELETED가 된 문서에는 저장하지 않고 DELETED를 반환한다."""
    document_id = make_ready_document(db_session, {1: "원문"})
    target = require_target(repository, document_id)
    delete_document(db_session, document_id, kind)

    assert store(repository, target, "조각") == ChunkIndexSkipReason.DELETED
    assert stored_rows(db_session, document_id) == []


@pytest.mark.parametrize("kind", DELETIONS)
def test_deleted_document_is_not_unindexed_candidate(
    db_session: Session, repository: SqlAlchemyDocumentChunkRepository, kind: str
) -> None:
    """deleted_at이 있거나 DELETED인 문서는 재처리 대상으로 고르지 않는다."""
    document_id = make_ready_document(db_session, {1: "원문"})
    delete_document(db_session, document_id, kind)

    assert document_id not in repository.find_unindexed_document_ids(
        model_name=MODEL_NAME, limit=10000
    )


def test_health_record_embeddings_are_untouched(
    db_session: Session, repository: SqlAlchemyDocumentChunkRepository
) -> None:
    """조각을 바꿔도 같은 환자의 HEALTH_RECORD 임베딩은 남는다."""
    document_id = make_ready_document(db_session, {1: "원문"})
    target = require_target(repository, document_id)
    health_id = insert_health_record_embedding(db_session, target.patient_id)

    store(repository, target, "조각")
    store(repository, target, "새 조각")

    assert (
        db_session.execute(
            text("SELECT count(*) FROM embeddings WHERE id = :id"), {"id": health_id}
        ).scalar_one()
        == 1
    )


def test_mismatched_vector_count_writes_nothing(
    db_session: Session, repository: SqlAlchemyDocumentChunkRepository
) -> None:
    """조각 수와 벡터 수가 다르면 아무것도 쓰지 않고 ValueError를 던진다."""
    document_id = make_ready_document(db_session, {1: "원문"})
    target = require_target(repository, document_id)

    with pytest.raises(ValueError):
        repository.replace_chunks(target, make_chunks(target, "조각"), [], MODEL_NAME)

    assert stored_rows(db_session, document_id) == []


def test_unindexed_documents_are_found(
    db_session: Session, repository: SqlAlchemyDocumentChunkRepository
) -> None:
    """임베딩이 없는 성공 문서만 고르고, 색인된 문서와 원문 없는 문서는 고르지 않는다."""
    unindexed = make_ready_document(db_session, {1: "원문"})
    indexed = make_ready_document(db_session, {1: "원문"})
    store(repository, require_target(repository, indexed), "조각")
    blank = make_ready_document(db_session, {1: None})
    running = make_ready_document(db_session, {1: "원문"})
    add_newer_analysis(db_session, running, "RUNNING")

    found = repository.find_unindexed_document_ids(model_name=MODEL_NAME, limit=10000)

    assert unindexed in found
    assert indexed not in found
    assert blank not in found
    assert running not in found


class FakeEmbedder:
    """입력 수만 기록하고 768차원 합성 벡터를 돌려주는 가짜 임베더. 외부 API를 부르지 않는다."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """호출을 기록하고 입력 수만큼 벡터를 만든다."""
        self.calls.append(list(texts))
        return vectors(len(texts))


def test_new_analysis_without_chunks_keeps_previous_index(
    db_session: Session, repository: SqlAlchemyDocumentChunkRepository
) -> None:
    """새 분석의 원문에서 조각이 나오지 않으면 이전 분석의 조각과 임베딩이 그대로 남는다."""
    document_id = make_ready_document(db_session, {1: "혈색소 13.2 g/dL"})
    embedder = FakeEmbedder()
    index_document_chunks(document_id, repository=repository, embedder=embedder)
    before = stored_rows(db_session, document_id)
    add_newer_analysis(db_session, document_id, "SUCCEEDED")
    db_session.execute(
        text("UPDATE document_pages SET ocr_text = '  ' WHERE document_id = :id"),
        {"id": document_id},
    )

    outcome = index_document_chunks(document_id, repository=repository, embedder=embedder)

    assert (outcome.status, outcome.reason) == (
        ChunkIndexStatus.SKIPPED,
        ChunkIndexSkipReason.NO_CHUNKS,
    )
    assert len(embedder.calls) == 1
    assert len(before) == 1
    assert stored_rows(db_session, document_id) == before
    assert count_chunk_embeddings(db_session, document_id) == 1


def chunk_ids(db_session: Session, document_id: UUID) -> list[UUID]:
    """문서의 조각 id를 조각 번호 순서로 읽는다."""
    return list(
        db_session.execute(
            text(
                "SELECT c.id FROM document_chunks c"
                " JOIN document_analyses a ON a.id = c.analysis_id"
                " WHERE a.document_id = :id ORDER BY c.chunk_index"
            ),
            {"id": document_id},
        ).scalars()
    )


def drop_embedding_of_chunk(db_session: Session, document_id: UUID, chunk_index: int) -> None:
    """문서의 chunk_index 조각에 달린 임베딩만 지워 부분 색인 상태를 만든다."""
    db_session.execute(
        text(
            "DELETE FROM embeddings e USING document_chunks c, document_analyses a"
            " WHERE e.source_type = 'DOCUMENT_CHUNK' AND e.source_id = c.id"
            " AND a.id = c.analysis_id AND a.document_id = :id AND c.chunk_index = :index"
        ),
        {"id": document_id, "index": chunk_index},
    )


def test_partially_indexed_document_is_refilled_once(
    db_session: Session, repository: SqlAlchemyDocumentChunkRepository
) -> None:
    """임베딩이 빠진 조각이 있으면 목록에 뜨고, 재색인은 빠진 조각만 임베딩한다.

    다시 실행하면 목록에서 빠지고 임베더를 부르지 않는다.
    """
    document_id = make_ready_document(db_session, {1: "원문"})
    store(repository, require_target(repository, document_id), "첫 조각", "둘째 조각", "셋째 조각")
    drop_embedding_of_chunk(db_session, document_id, 1)
    chunk_ids_before = chunk_ids(db_session, document_id)

    assert document_id in repository.find_unindexed_document_ids(model_name=MODEL_NAME, limit=10000)

    embedder = FakeEmbedder()
    outcome = index_document_chunks(document_id, repository=repository, embedder=embedder)

    assert embedder.calls == [["둘째 조각"]]
    assert (outcome.status, outcome.chunk_count, outcome.embedded_count) == (
        ChunkIndexStatus.INDEXED,
        3,
        1,
    )
    rows = stored_rows(db_session, document_id)
    # 조각 행은 지우지 않고 그대로 둔다.
    assert chunk_ids(db_session, document_id) == chunk_ids_before
    assert all(row.model_name == MODEL_NAME for row in rows)
    assert count_chunk_embeddings(db_session, document_id) == 3
    assert document_id not in repository.find_unindexed_document_ids(
        model_name=MODEL_NAME, limit=10000
    )

    again = index_document_chunks(document_id, repository=repository, embedder=embedder)

    assert len(embedder.calls) == 1
    assert (again.status, again.embedded_count) == (ChunkIndexStatus.INDEXED, 0)


def test_other_model_embeddings_are_not_indexed(
    db_session: Session, repository: SqlAlchemyDocumentChunkRepository
) -> None:
    """다른 모델의 임베딩만 있으면 목록에 뜨고, 채우면 조각마다 현재 모델 임베딩 하나로 바뀐다."""
    document_id = make_ready_document(db_session, {1: "원문"})
    store(repository, require_target(repository, document_id), "첫 조각", "둘째 조각")

    assert document_id in repository.find_unindexed_document_ids(
        model_name="other-model", limit=10000
    )

    embedder = FakeEmbedder("other-model")
    index_document_chunks(document_id, repository=repository, embedder=embedder)

    assert embedder.calls == [["첫 조각", "둘째 조각"]]
    assert [row.model_name for row in stored_rows(db_session, document_id)] == [
        "other-model",
        "other-model",
    ]
    assert count_chunk_embeddings(db_session, document_id) == 2


def test_fill_skips_chunks_filled_meanwhile(
    db_session: Session, repository: SqlAlchemyDocumentChunkRepository
) -> None:
    """읽은 뒤 다른 실행이 채운 조각에는 임베딩을 겹쳐 넣지 않는다."""
    document_id = make_ready_document(db_session, {1: "원문"})
    store(repository, require_target(repository, document_id), "첫 조각")
    drop_embedding_of_chunk(db_session, document_id, 0)
    target = require_target(repository, document_id)
    [missing] = target.missing_chunks
    # 다른 실행이 먼저 채운 상황.
    assert (
        repository.fill_missing_embeddings(target, [missing.chunk_id], vectors(1), MODEL_NAME)
        is None
    )

    assert (
        repository.fill_missing_embeddings(target, [missing.chunk_id], vectors(1), MODEL_NAME)
        is None
    )
    assert count_chunk_embeddings(db_session, document_id) == 1


def test_fill_for_superseded_analysis_changes_nothing(
    db_session: Session, repository: SqlAlchemyDocumentChunkRepository
) -> None:
    """채우는 사이 새 분석이 생겼으면 아무것도 넣지 않고 STALE_ANALYSIS를 반환한다."""
    document_id = make_ready_document(db_session, {1: "원문"})
    store(repository, require_target(repository, document_id), "첫 조각")
    drop_embedding_of_chunk(db_session, document_id, 0)
    target = require_target(repository, document_id)
    add_newer_analysis(db_session, document_id, "SUCCEEDED")

    reason = repository.fill_missing_embeddings(
        target, [target.missing_chunks[0].chunk_id], vectors(1), MODEL_NAME
    )

    assert reason == ChunkIndexSkipReason.STALE_ANALYSIS
    assert count_chunk_embeddings(db_session, document_id) == 0


def count_chunk_embeddings(db_session: Session, document_id: UUID) -> int:
    """문서의 모든 분석에 달린 DOCUMENT_CHUNK 임베딩 수를 센다. 고아 임베딩도 센다."""
    return db_session.execute(
        text(
            "SELECT count(*) FROM embeddings e WHERE e.source_type = 'DOCUMENT_CHUNK'"
            " AND e.patient_id = (SELECT patient_id FROM documents WHERE id = :id)"
        ),
        {"id": document_id},
    ).scalar_one()


def cite_first_chunk(db_session: Session, document_id: UUID) -> None:
    """설명 인용(explanation_citations)이 문서의 첫 조각을 가리키게 한다."""
    explanation_id = uuid4()
    db_session.execute(
        text("INSERT INTO ai_explanations (id, document_id) VALUES (:id, :document_id)"),
        {"id": explanation_id, "document_id": document_id},
    )
    db_session.execute(
        text(
            "INSERT INTO explanation_citations (id, explanation_id, chunk_id)"
            " SELECT :id, :explanation_id, c.id FROM document_chunks c"
            " JOIN document_analyses a ON a.id = c.analysis_id"
            " WHERE a.document_id = :document_id AND c.chunk_index = 0"
        ),
        {"id": uuid4(), "explanation_id": explanation_id, "document_id": document_id},
    )


def insert_health_record_embedding(db_session: Session, patient_id: UUID) -> UUID:
    """같은 환자의 HEALTH_RECORD 임베딩 한 행을 넣는다."""
    embedding_id = uuid4()
    db_session.execute(
        text(
            "INSERT INTO embeddings (id, source_type, source_id, patient_id, embedding, model_name)"
            " VALUES (:id, 'HEALTH_RECORD', :source_id, :patient_id,"
            " CAST(:embedding AS vector), 'test')"
        ),
        {
            "id": embedding_id,
            "source_id": uuid4(),
            "patient_id": patient_id,
            "embedding": "[" + ",".join(["0.1"] * DIMENSIONS) + "]",
        },
    )
    return embedding_id
