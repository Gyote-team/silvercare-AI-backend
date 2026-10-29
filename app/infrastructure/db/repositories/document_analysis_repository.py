"""분석 상태와 추출 결과를 documents, document_analyses, document_pages, extracted_items에 쓴다."""

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import delete, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.application.ports.document_analysis_repository import AnalysisTarget
from app.domain.document.document_analysis import (
    AnalysisFailure,
    AnalysisStatus,
    DocumentStatus,
)
from app.domain.document.document_page import PageOcrStatus, RenderedPage
from app.domain.document.page_extraction import PageExtraction
from app.infrastructure.db.models.document import DocumentModel
from app.infrastructure.db.models.document_analysis import DocumentAnalysisModel
from app.infrastructure.db.models.document_page import DocumentPageModel
from app.infrastructure.db.models.extracted_item import ExtractedItemModel
from app.infrastructure.db.repositories.document_page_repository import (
    SqlAlchemyDocumentPageRepository,
)

logger = logging.getLogger(__name__)

_NOT_DELETED = (
    DocumentModel.status != DocumentStatus.DELETED,
    DocumentModel.deleted_at.is_(None),
)


class SqlAlchemyDocumentAnalysisRepository:
    """메서드마다 session을 열고 끝날 때 commit한다. 메서드 하나가 트랜잭션 하나다.

    DocumentAnalysisRepository와 DocumentPageRepository 포트를 함께 구현한다.
    stale_after는 RUNNING 분석을 멈춘 것으로 보고 다시 시작을 허용하기까지의 시간이다.
    """

    def __init__(self, session_factory: sessionmaker[Session], *, stale_after: timedelta) -> None:
        self._session_factory = session_factory
        self._stale_after = stale_after

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
        """문서를 PROCESSING, 분석을 RUNNING으로 바꾸고 이전 실행의 결과 컬럼을 비운다.

        분석이 SUCCEEDED이거나, RUNNING이고 시작한 지 stale_after가 지나지 않았으면
        아무것도 바꾸지 않고 False를 반환한다.
        """
        with self._session_factory.begin() as session:
            # 행 잠금 안에서 판단해야 동시 요청 중 하나만 시작한다.
            analysis = session.execute(
                select(DocumentAnalysisModel.status, DocumentAnalysisModel.started_at)
                .where(DocumentAnalysisModel.id == target.analysis_id)
                .with_for_update()
            ).one()
            # 끝난 분석은 다시 돌리지 않는다. 재분석은 새 분석 행으로 한다(FC-08-03-02).
            if analysis.status == AnalysisStatus.SUCCEEDED:
                return False
            # 시작 시각이 없는 RUNNING은 만료를 판단할 수 없어 멈춘 것으로 본다.
            if (
                analysis.status == AnalysisStatus.RUNNING
                and analysis.started_at is not None
                and started_at - analysis.started_at < self._stale_after
            ):
                return False
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
        started_at: datetime,
        document_status: DocumentStatus,
        extractions_by_page_no: Mapping[int, PageExtraction],
        model_version: str | None,
        completed_at: datetime,
    ) -> bool:
        """추출 항목 교체, 페이지 원문 기록, 분석과 문서 상태 변경을 한 트랜잭션으로 묶는다.

        이 실행이 분석을 맡고 있지 않으면(RUNNING이 아니거나 started_at이 다르면) False를 반환한다.
        """
        with self._session_factory.begin() as session:
            if not _lock_own_run(session, target, started_at):
                return False
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
                for page_no, extraction in extractions_by_page_no.items()
                for item in extraction.items
            ]
            if rows:
                session.execute(insert(ExtractedItemModel), rows)
            # 모든 페이지를 실패로 둔 뒤 성공한 페이지만 덮어쓴다.
            _mark_pages_failed(session, target)
            page_rows = [
                {
                    "id": page_ids[page_no],
                    "ocr_status": (
                        PageOcrStatus.SUCCEEDED if extraction.has_text else PageOcrStatus.SKIPPED
                    ),
                    "ocr_text": extraction.raw_text if extraction.has_text else None,
                }
                for page_no, extraction in extractions_by_page_no.items()
            ]
            if page_rows:
                session.execute(update(DocumentPageModel), page_rows)
            _update_analysis(
                session,
                target,
                status=AnalysisStatus.SUCCEEDED,
                completed_at=completed_at,
                model_version=model_version,
            )
            return True

    def mark_failed(
        self,
        target: AnalysisTarget,
        started_at: datetime,
        failure: AnalysisFailure,
        completed_at: datetime,
    ) -> bool:
        """추출 항목 삭제, 페이지 FAILED, 실패 기록, 문서와 분석 FAILED를 한 트랜잭션으로 묶는다.

        설명이 참조해 항목을 지울 수 없으면 항목은 남기고 나머지만 기록한다.
        이 실행이 분석을 맡고 있지 않으면 save_result와 같이 False를 반환한다.
        """
        with self._session_factory.begin() as session:
            if not _lock_own_run(session, target, started_at):
                return False
            if not _update_document(session, target, DocumentStatus.FAILED, completed_at):
                return False
            try:
                with session.begin_nested():
                    _delete_items(session, target)
            except IntegrityError as error:
                # 삭제만 되돌리고 실패 기록은 남겨야 PROCESSING/RUNNING에 멈추지 않는다.
                logger.warning(
                    "추출 항목을 지우지 못해 남겨 둡니다: document_id=%s, analysis_id=%s, error=%s",
                    target.document_id,
                    target.analysis_id,
                    type(error).__name__,
                )
            _mark_pages_failed(session, target)
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

    def mark_failed_before_start(
        self, document_id: UUID, failure: AnalysisFailure, completed_at: datetime
    ) -> bool:
        """시작 전에 실패한 문서의 최신 분석이 PENDING이면 문서와 분석을 FAILED로 기록한다.

        이미 다른 상태면 다른 실행이 맡았거나 끝난 분석이므로 아무것도 바꾸지 않고 False를 반환한다.
        이번 실행이 만든 페이지와 항목이 없으므로 둘은 건드리지 않는다.
        """
        with self._session_factory.begin() as session:
            # 최신 분석을 고르는 기준은 find_target과 같다.
            analysis = session.execute(
                select(DocumentAnalysisModel.id, DocumentAnalysisModel.status)
                .where(DocumentAnalysisModel.document_id == document_id)
                .order_by(DocumentAnalysisModel.created_at.desc(), DocumentAnalysisModel.id.desc())
                .limit(1)
                .with_for_update()
            ).one_or_none()
            if analysis is None or analysis.status != AnalysisStatus.PENDING:
                return False
            # 그 사이 삭제된 문서는 덮어쓰지 않는다. _update_document와 같은 조건이다.
            document = session.execute(
                update(DocumentModel)
                .where(DocumentModel.id == document_id, *_NOT_DELETED)
                .values(
                    status=DocumentStatus.FAILED,
                    status_changed_at=completed_at,
                    updated_at=completed_at,
                )
            )
            if document.rowcount != 1:
                return False
            session.execute(
                update(DocumentAnalysisModel)
                .where(DocumentAnalysisModel.id == analysis.id)
                .values(
                    status=AnalysisStatus.FAILED,
                    completed_at=completed_at,
                    error_code=failure.error_code,
                    failed_step=failure.failed_step,
                    error_message=failure.detail_code,
                    retryable=failure.retryable,
                )
            )
            return True


def _lock_own_run(session: Session, target: AnalysisTarget, started_at: datetime) -> bool:
    # 잠근 채로 확인해야 그사이 만료 재시작(mark_started)이 끼어들지 못한다.
    owned = session.execute(
        select(DocumentAnalysisModel.id)
        .where(
            DocumentAnalysisModel.id == target.analysis_id,
            DocumentAnalysisModel.status == AnalysisStatus.RUNNING,
            DocumentAnalysisModel.started_at == started_at,
        )
        .with_for_update()
    ).scalar_one_or_none()
    if owned is None:
        logger.info(
            "다른 실행이 맡은 분석이라 저장하지 않습니다: document_id=%s, analysis_id=%s",
            target.document_id,
            target.analysis_id,
        )
        return False
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


def _mark_pages_failed(session: Session, target: AnalysisTarget) -> None:
    # 이전 실행의 원문이 실패한 페이지에 남지 않게 ocr_text도 비운다.
    session.execute(
        update(DocumentPageModel)
        .where(DocumentPageModel.document_id == target.document_id)
        .values(ocr_status=PageOcrStatus.FAILED, ocr_text=None)
    )
