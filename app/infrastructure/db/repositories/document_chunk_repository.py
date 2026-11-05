"""문서 조각을 document_chunks에, 그 임베딩을 embeddings(DOCUMENT_CHUNK)에 쓴다."""

from collections.abc import Sequence
from uuid import UUID, uuid4

from sqlalchemy import Row, Select, column, delete, exists, func, insert, select, table, text
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.orm import Session, sessionmaker

from app.application.ports.document_chunk_repository import ChunkIndexSkipReason, ChunkIndexTarget
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
_EMBEDDINGS = table("embeddings", column("source_type"), column("source_id"))
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

    def find_index_target(self, document_id: UUID) -> ChunkIndexTarget | ChunkIndexSkipReason:
        """READY·NEEDS_REVIEW 문서의 최신 성공 분석과 원문이 있는 페이지를 읽는다."""
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
        return ChunkIndexTarget(
            document_id=document_id,
            analysis_id=analysis.id,
            patient_id=document.patient_id,
            visit_id=document.visit_id,
            pages=tuple(PageText(page.id, page.page_no, page.ocr_text) for page in pages),
            failed_page_count=failed_page_count,
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
            # 문서 행을 잠가 같은 문서의 교체와 삭제가 겹치지 않게 한다.
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
            _delete_document_chunks(session, target.document_id)
            _insert_chunks(session, target, chunks, vectors, model_name)
        return None

    def find_unindexed_document_ids(self, *, limit: int) -> list[UUID]:
        """분석은 성공했지만 최신 분석의 조각 임베딩이 없는 문서 id를 오래된 분석부터 읽는다.

        원문이 있는 SUCCEEDED 페이지가 없는 문서는 조각이 생기지 않으므로 고르지 않는다.
        """
        latest = _latest_analyses().subquery()
        has_text = exists().where(
            DocumentPageModel.document_id == latest.c.document_id,
            DocumentPageModel.ocr_status == PageOcrStatus.SUCCEEDED,
            DocumentPageModel.ocr_text.is_not(None),
        )
        has_embedding = (
            select(DocumentChunkModel.id)
            .join(_EMBEDDINGS, _EMBEDDINGS.c.source_id == DocumentChunkModel.id)
            .where(
                DocumentChunkModel.analysis_id == latest.c.id,
                _EMBEDDINGS.c.source_type == SOURCE_TYPE,
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
                    has_text,
                    ~has_embedding,
                )
                .order_by(latest.c.created_at)
                .limit(limit)
            ).scalars()
            return list(document_ids)


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
    session: Session,
    target: ChunkIndexTarget,
    chunks: Sequence[DocumentChunk],
    vectors: Sequence[Sequence[float]],
    model_name: str,
) -> None:
    if not chunks:
        return
    chunk_ids = [uuid4() for _ in chunks]
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
