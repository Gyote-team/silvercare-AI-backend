"""분석 상태와 추출 결과를 documents, document_analyses, extracted_items에 쓰는 SQLAlchemy 구현."""

from collections.abc import Mapping, Sequence
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import delete, insert, select, update
from sqlalchemy.orm import Session, sessionmaker

from app.application.ports.document_analysis_repository import AnalysisTarget
from app.domain.document.document_analysis import (
    AnalysisFailure,
    AnalysisStatus,
    DocumentStatus,
)
from app.domain.document.document_page import RenderedPage
from app.domain.document.page_extraction import ExtractedItem
from app.infrastructure.db.models.document import DocumentModel
from app.infrastructure.db.models.document_analysis import DocumentAnalysisModel
from app.infrastructure.db.models.document_page import DocumentPageModel
from app.infrastructure.db.models.extracted_item import ExtractedItemModel
from app.infrastructure.db.repositories.document_page_repository import (
    SqlAlchemyDocumentPageRepository,
)

_NOT_DELETED = (
    DocumentModel.status != DocumentStatus.DELETED,
    DocumentModel.deleted_at.is_(None),
)


class SqlAlchemyDocumentAnalysisRepository:
    """메서드마다 session을 열고 끝날 때 commit한다. 메서드 하나가 트랜잭션 하나다.

    DocumentAnalysisRepository와 DocumentPageRepository 포트를 함께 구현한다.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def find_target(self, document_id: UUID) -> AnalysisTarget | None:
        """삭제되지 않은 문서와 그 문서의 최신 분석 작업을 읽는다."""
        with self._session_factory() as session:
            document = session.execute(
                select(DocumentModel.storage_key, DocumentModel.mime_type).where(
                    DocumentModel.id == document_id, *_NOT_DELETED
                )
            ).one_or_none()
            if document is None:
                return None
            # 최신 분석을 고르는 기준은 Spring 목록 쿼리와 같다.
            analysis_id = session.execute(
                select(DocumentAnalysisModel.id)
                .where(DocumentAnalysisModel.document_id == document_id)
                .order_by(DocumentAnalysisModel.created_at.desc(), DocumentAnalysisModel.id.desc())
                .limit(1)
            ).scalar_one_or_none()
        if analysis_id is None:
            return None
        return AnalysisTarget(
            document_id=document_id,
            analysis_id=analysis_id,
            storage_key=document.storage_key,
            mime_type=document.mime_type,
        )

    def mark_started(self, target: AnalysisTarget, started_at: datetime) -> bool:
        """문서를 PROCESSING, 분석을 RUNNING으로 바꾸고 이전 실행의 결과 컬럼을 비운다."""
        with self._session_factory.begin() as session:
            if not _update_document(session, target, DocumentStatus.PROCESSING, started_at):
                return False
            _update_analysis(
                session,
                target,
                status=AnalysisStatus.RUNNING,
                started_at=started_at,
                completed_at=None,
                model_version=None,
                error_code=None,
                failed_step=None,
                error_message=None,
                retryable=False,
            )
            return True

    def save_rendered_pages(
        self, document_id: UUID, pages: Sequence[RenderedPage], rendered_at: datetime
    ) -> None:
        """document_pages 저장을 한 트랜잭션으로 commit한다."""
        with self._session_factory.begin() as session:
            SqlAlchemyDocumentPageRepository(session).save_rendered_pages(
                document_id, pages, rendered_at
            )

    def save_result(
        self,
        target: AnalysisTarget,
        document_status: DocumentStatus,
        items_by_page_no: Mapping[int, Sequence[ExtractedItem]],
        model_version: str | None,
        completed_at: datetime,
    ) -> bool:
        """추출 항목 교체, 분석 SUCCEEDED, 문서 상태 변경을 한 트랜잭션으로 묶는다."""
        with self._session_factory.begin() as session:
            if not _update_document(session, target, document_status, completed_at):
                return False
            _delete_items(session, target)
            page_ids = dict(
                session.execute(
                    select(DocumentPageModel.page_no, DocumentPageModel.id).where(
                        DocumentPageModel.document_id == target.document_id
                    )
                ).all()
            )
            rows = [
                {
                    **item.model_dump(mode="json"),
                    "id": uuid4(),
                    "analysis_id": target.analysis_id,
                    "page_id": page_ids[page_no],
                }
                for page_no, items in items_by_page_no.items()
                for item in items
            ]
            if rows:
                session.execute(insert(ExtractedItemModel), rows)
            _update_analysis(
                session,
                target,
                status=AnalysisStatus.SUCCEEDED,
                completed_at=completed_at,
                model_version=model_version,
            )
            return True

    def mark_failed(
        self, target: AnalysisTarget, failure: AnalysisFailure, completed_at: datetime
    ) -> bool:
        """추출 항목 삭제, 실패 기록, 문서와 분석 FAILED를 한 트랜잭션으로 묶는다."""
        with self._session_factory.begin() as session:
            if not _update_document(session, target, DocumentStatus.FAILED, completed_at):
                return False
            _delete_items(session, target)
            _update_analysis(
                session,
                target,
                status=AnalysisStatus.FAILED,
                completed_at=completed_at,
                error_code=failure.error_code,
                failed_step=failure.failed_step,
                error_message=failure.detail_code,
                retryable=failure.retryable,
            )
            return True


def _update_document(
    session: Session, target: AnalysisTarget, status: DocumentStatus, changed_at: datetime
) -> bool:
    # 그 사이 삭제된 문서를 덮어쓰지 않게 조건을 DB에서 확인한다. Spring과 같은 방식이다.
    result = session.execute(
        update(DocumentModel)
        .where(DocumentModel.id == target.document_id, *_NOT_DELETED)
        .values(status=status, status_changed_at=changed_at, updated_at=changed_at)
    )
    return result.rowcount == 1


def _update_analysis(session: Session, target: AnalysisTarget, **values: object) -> None:
    session.execute(
        update(DocumentAnalysisModel)
        .where(DocumentAnalysisModel.id == target.analysis_id)
        .values(**values)
    )


def _delete_items(session: Session, target: AnalysisTarget) -> None:
    # 유일 키가 없어 upsert를 쓸 수 없으므로, 다시 실행해도 쌓이지 않게 지우고 넣는다.
    session.execute(
        delete(ExtractedItemModel).where(ExtractedItemModel.analysis_id == target.analysis_id)
    )
