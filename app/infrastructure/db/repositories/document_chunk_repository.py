"""문서 조각을 document_chunks에, 그 임베딩을 embeddings(DOCUMENT_CHUNK)에 쓴다."""

from collections.abc import Sequence
from uuid import UUID, uuid4

from sqlalchemy import (
    Exists,
    Row,
    Select,
    and_,
    column,
    delete,
    exists,
    func,
    insert,
    or_,
    select,
    table,
    text,
)
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.orm import Session, sessionmaker

from app.application.ports.document_chunk_repository import (
    ChunkIndexSkipReason,
    ChunkIndexTarget,
    StoredChunk,
)
from app.application.ports.text_embedder import EMBEDDING_DIMENSIONS
from app.domain.document.document_analysis import AnalysisStatus, DocumentStatus
from app.domain.document.document_chunk import DocumentChunk, PageText
from app.domain.document.document_page import PageOcrStatus
from app.infrastructure.db.models.document import DocumentModel
from app.infrastructure.db.models.document_analysis import DocumentAnalysisModel
from app.infrastructure.db.models.document_chunk import DocumentChunkModel
from app.infrastructure.db.models.document_page import DocumentPageModel

SOURCE_TYPE = "DOCUMENT_CHUNK"

# 분석이 끝나 조각을 만들 수 있는 문서. 이 상태면 삭제(DELETED)가 아니다.
_INDEXABLE = (
    DocumentModel.status.in_((DocumentStatus.READY, DocumentStatus.NEEDS_REVIEW)),
    DocumentModel.deleted_at.is_(None),
)
# embeddings는 vector 칼럼이 있어 ORM 매핑 없이 지우기·조회에 쓰는 칼럼만 적는다.
_EMBEDDINGS = table(
    "embeddings",
    column("source_type"),
    column("source_id"),
    column("model_name"),
    column("dimensions"),
)
# vector 값은 pgvector 패키지 없이 '[1.0,2.0]' 문자열을 CAST로 넣는다.
_INSERT_EMBEDDING = text(
    "INSERT INTO embeddings"
    " (id, source_type, source_id, patient_id, visit_id, embedding, model_name, dimensions)"
    " VALUES (:id, :source_type, :source_id, :patient_id, :visit_id,"
    " CAST(:embedding AS vector), :model_name, :dimensions)"
)


class SqlAlchemyDocumentChunkRepository:
    """메서드마다 session을 열고 끝날 때 commit한다. 메서드 하나가 트랜잭션 하나다."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def find_index_target(
        self, document_id: UUID, model_name: str
    ) -> ChunkIndexTarget | ChunkIndexSkipReason:
        """READY·NEEDS_REVIEW 문서의 최신 성공 분석, 원문이 있는 페이지, 저장된 조각을 읽는다."""
        with self._session_factory() as session:
            document = session.execute(
                select(
                    DocumentModel.patient_id,
                    DocumentModel.visit_id,
                    DocumentModel.status,
                    DocumentModel.deleted_at,
                ).where(DocumentModel.id == document_id)
            ).one_or_none()
            if (
                document is None
                or document.deleted_at is not None
                or document.status == DocumentStatus.DELETED
            ):
                return ChunkIndexSkipReason.DELETED
            if document.status not in (DocumentStatus.READY, DocumentStatus.NEEDS_REVIEW):
                return ChunkIndexSkipReason.NOT_ANALYZED
            analysis = _latest_analysis(session, document_id)
            if analysis is None or analysis.status != AnalysisStatus.SUCCEEDED:
                return ChunkIndexSkipReason.NOT_ANALYZED
            pages = session.execute(
                select(DocumentPageModel.id, DocumentPageModel.page_no, DocumentPageModel.ocr_text)
                .where(
                    DocumentPageModel.document_id == document_id,
                    DocumentPageModel.ocr_status == PageOcrStatus.SUCCEEDED,
                    DocumentPageModel.ocr_text.is_not(None),
                )
                .order_by(DocumentPageModel.page_no)
            ).all()
            failed_page_count = session.execute(
                select(func.count()).where(
                    DocumentPageModel.document_id == document_id,
                    DocumentPageModel.ocr_status == PageOcrStatus.FAILED,
                )
            ).scalar_one()
            stored = session.execute(
                select(
                    DocumentChunkModel.id,
                    DocumentChunkModel.content,
                    _has_current_embedding(model_name).label("embedded"),
                )
                .where(DocumentChunkModel.analysis_id == analysis.id)
                .order_by(DocumentChunkModel.chunk_index)
            ).all()
        return ChunkIndexTarget(
            document_id=document_id,
            analysis_id=analysis.id,
            patient_id=document.patient_id,
            visit_id=document.visit_id,
            pages=tuple(PageText(page.id, page.page_no, page.ocr_text) for page in pages),
            failed_page_count=failed_page_count,
            stored_chunk_count=len(stored),
            missing_chunks=tuple(
                StoredChunk(row.id, row.content) for row in stored if not row.embedded
            ),
        )

    def replace_chunks(
        self,
        target: ChunkIndexTarget,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
        model_name: str,
    ) -> ChunkIndexSkipReason | None:
        """이 문서의 이전 조각·임베딩을 지우고 새 것을 한 트랜잭션에서 넣는다.

        같은 분석을 다시 실행해도 (analysis_id, chunk_index) 유일 키에 걸리지 않게 먼저 지운다.
        이전 조각이 인용돼 FK에 막히면 예외가 나고 전부 롤백되어 이전 것이 남는다.
        """
        if len(chunks) != len(vectors):
            raise ValueError("조각 수와 벡터 수가 다릅니다")
        with self._session_factory.begin() as session:
            reason = _lock_latest_target(session, target)
            if reason is not None:
                return reason
            _delete_document_chunks(session, target.document_id)
            chunk_ids = _insert_chunks(session, target, chunks)
            _insert_embeddings(session, target, chunk_ids, vectors, model_name)
        return None

    def fill_missing_embeddings(
        self,
        target: ChunkIndexTarget,
        chunk_ids: Sequence[UUID],
        vectors: Sequence[Sequence[float]],
        model_name: str,
    ) -> ChunkIndexSkipReason | None:
        """target 분석의 조각 중 아직 현재 기준 임베딩이 없는 것에만 임베딩을 넣는다."""
        if len(chunk_ids) != len(vectors):
            raise ValueError("조각 수와 벡터 수가 다릅니다")
        with self._session_factory.begin() as session:
            reason = _lock_latest_target(session, target)
            if reason is not None:
                return reason
            # 잠근 뒤 다시 골라, 그사이 다른 실행이 채운 조각에 같은 임베딩이 겹치지 않게 한다.
            still_missing = set(
                session.execute(
                    select(DocumentChunkModel.id).where(
                        DocumentChunkModel.id.in_(chunk_ids),
                        DocumentChunkModel.analysis_id == target.analysis_id,
                        ~_has_current_embedding(model_name),
                    )
                ).scalars()
            )
            if not still_missing:
                return None
            # 남은 임베딩은 다른 모델·차원 것뿐이므로 지우고 현재 기준으로 바꾼다.
            session.execute(
                delete(_EMBEDDINGS).where(
                    _EMBEDDINGS.c.source_type == SOURCE_TYPE,
                    _EMBEDDINGS.c.source_id.in_(still_missing),
                )
            )
            pairs = [
                (chunk_id, vector)
                for chunk_id, vector in zip(chunk_ids, vectors, strict=True)
                if chunk_id in still_missing
            ]
            _insert_embeddings(
                session,
                target,
                [chunk_id for chunk_id, _ in pairs],
                [vector for _, vector in pairs],
                model_name,
            )
        return None

    def find_unindexed_document_ids(self, *, model_name: str, limit: int) -> list[UUID]:
        """분석은 성공했지만 최신 분석이 model_name 기준으로 색인되지 않은 문서 id를 읽는다.

        오래된 분석부터 고른다. 최신 분석의 조각이 없으면 원문이 있는 SUCCEEDED 페이지가 있을
        때만, 조각이 있으면 현재 기준 임베딩이 없는 조각이 하나라도 있을 때 고른다.
        """
        latest = _latest_analyses().subquery()
        has_text = exists().where(
            DocumentPageModel.document_id == latest.c.document_id,
            DocumentPageModel.ocr_status == PageOcrStatus.SUCCEEDED,
            DocumentPageModel.ocr_text.is_not(None),
        )
        has_chunks = exists().where(DocumentChunkModel.analysis_id == latest.c.id)
        has_missing = (
            select(DocumentChunkModel.id)
            .where(
                DocumentChunkModel.analysis_id == latest.c.id,
                ~_has_current_embedding(model_name),
            )
            .exists()
        )
        with self._session_factory() as session:
            document_ids = session.execute(
                select(latest.c.document_id)
                .join(DocumentModel, DocumentModel.id == latest.c.document_id)
                .where(
                    *_INDEXABLE,
                    latest.c.status == AnalysisStatus.SUCCEEDED,
                    or_(and_(~has_chunks, has_text), has_missing),
                )
                .order_by(latest.c.created_at)
                .limit(limit)
            ).scalars()
            return list(document_ids)


def _has_current_embedding(model_name: str) -> Exists:
    # 바깥 쿼리의 조각 행에 현재 모델·차원의 임베딩이 있는지. 다른 모델 것은 색인으로 보지 않는다.
    return exists().where(
        _EMBEDDINGS.c.source_type == SOURCE_TYPE,
        _EMBEDDINGS.c.source_id == DocumentChunkModel.id,
        _EMBEDDINGS.c.model_name == model_name,
        _EMBEDDINGS.c.dimensions == EMBEDDING_DIMENSIONS,
    )


def _lock_latest_target(session: Session, target: ChunkIndexTarget) -> ChunkIndexSkipReason | None:
    # 문서 행을 잠가 같은 문서의 교체·채우기와 삭제가 겹치지 않게 하고, 잠금 안에서 최신인지 본다.
    locked = session.execute(
        select(DocumentModel.id)
        .where(DocumentModel.id == target.document_id, *_INDEXABLE)
        .with_for_update()
    ).scalar_one_or_none()
    if locked is None:
        return ChunkIndexSkipReason.DELETED
    latest = _latest_analysis(session, target.document_id)
    if (
        latest is None
        or latest.id != target.analysis_id
        or latest.status != AnalysisStatus.SUCCEEDED
    ):
        return ChunkIndexSkipReason.STALE_ANALYSIS
    return None


def _latest_analyses() -> Select:
    # 문서마다 최신 분석 한 건. 고르는 기준은 분석 저장소의 find_target과 같다.
    return (
        select(
            DocumentAnalysisModel.id,
            DocumentAnalysisModel.document_id,
            DocumentAnalysisModel.status,
            DocumentAnalysisModel.created_at,
        )
        .ext(distinct_on(DocumentAnalysisModel.document_id))
        .order_by(
            DocumentAnalysisModel.document_id,
            DocumentAnalysisModel.created_at.desc(),
            DocumentAnalysisModel.id.desc(),
        )
    )


def _latest_analysis(session: Session, document_id: UUID) -> Row | None:
    return session.execute(
        _latest_analyses().where(DocumentAnalysisModel.document_id == document_id)
    ).one_or_none()


def _delete_document_chunks(session: Session, document_id: UUID) -> None:
    # 같은 분석 재실행과 새 분석 행 재분석을 함께 다루려고 문서의 모든 분석을 대상으로 한다.
    analysis_ids = select(DocumentAnalysisModel.id).where(
        DocumentAnalysisModel.document_id == document_id
    )
    chunk_ids = select(DocumentChunkModel.id).where(
        DocumentChunkModel.analysis_id.in_(analysis_ids)
    )
    session.execute(
        delete(_EMBEDDINGS).where(
            _EMBEDDINGS.c.source_type == SOURCE_TYPE, _EMBEDDINGS.c.source_id.in_(chunk_ids)
        )
    )
    session.execute(
        delete(DocumentChunkModel).where(DocumentChunkModel.analysis_id.in_(analysis_ids))
    )


def _insert_chunks(
    session: Session, target: ChunkIndexTarget, chunks: Sequence[DocumentChunk]
) -> list[UUID]:
    chunk_ids = [uuid4() for _ in chunks]
    if chunks:
        session.execute(
            insert(DocumentChunkModel),
            [
                {
                    "id": chunk_id,
                    "analysis_id": target.analysis_id,
                    "page_id": chunk.page_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "token_count": chunk.token_count,
                }
                for chunk_id, chunk in zip(chunk_ids, chunks, strict=True)
            ],
        )
    return chunk_ids


def _insert_embeddings(
    session: Session,
    target: ChunkIndexTarget,
    chunk_ids: Sequence[UUID],
    vectors: Sequence[Sequence[float]],
    model_name: str,
) -> None:
    if not chunk_ids:
        return
    session.execute(
        _INSERT_EMBEDDING,
        [
            {
                "id": uuid4(),
                "source_type": SOURCE_TYPE,
                "source_id": chunk_id,
                "patient_id": target.patient_id,
                "visit_id": target.visit_id,
                "embedding": "[" + ",".join(str(value) for value in vector) + "]",
                "model_name": model_name,
                "dimensions": len(vector),
            }
            for chunk_id, vector in zip(chunk_ids, vectors, strict=True)
        ],
    )
