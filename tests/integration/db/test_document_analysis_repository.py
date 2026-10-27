"""분석 상태·추출 결과 저장소 테스트. 실제 PostgreSQL에 쓰고 테스트가 끝나면 롤백한다."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Row, text
from sqlalchemy.orm import Session, sessionmaker

from app.application.ports.document_analysis_repository import AnalysisTarget
from app.domain.document.document_analysis import (
    AnalysisFailure,
    AnalysisStatus,
    DocumentStatus,
)
from app.domain.document.document_page import PageOcrStatus, RenderedPage, rendered_page_key
from app.domain.document.page_extraction import (
    DocumentType,
    ExtractedItem,
    ExtractedItemType,
    PageExtraction,
    SourceBox,
)
from app.domain.operation.error_code import AnalysisErrorCode, FailedStep
from app.infrastructure.db.repositories.document_analysis_repository import (
    SqlAlchemyDocumentAnalysisRepository,
)

STARTED_AT = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
COMPLETED_AT = datetime(2026, 10, 1, 9, 5, tzinfo=UTC)
STALE_AFTER = timedelta(minutes=60)
BOX = SourceBox(x=10, y=20, width=30, height=40)
FAILURE = AnalysisFailure(
    error_code=AnalysisErrorCode.VLM_TIMEOUT,
    failed_step=FailedStep.VLM_EXTRACTION,
    retryable=True,
    detail_code="TIMEOUT",
)
BLANK_PAGE = PageExtraction(document_type=DocumentType.UNKNOWN, raw_text="", items=[])
UNTOUCHED_PAGES = [(1, PageOcrStatus.PENDING, None), (2, PageOcrStatus.PENDING, None)]


def make_item(source_text: str, source_box: SourceBox | None = BOX) -> ExtractedItem:
    """합성 검사값 항목 하나를 만든다."""
    return ExtractedItem(
        item_type=ExtractedItemType.LAB_VALUE,
        normalized_name="혈색소",
        raw_value="13.5",
        normalized_value="13.5",
        unit="g/dL",
        reference_range="12.0-16.0",
        confidence=0.9,
        source_text=source_text,
        source_box=source_box,
    )


def make_extraction(*items: ExtractedItem, raw_text: str = "합성 원문") -> PageExtraction:
    """받은 항목과 원문이 든 페이지 추출 결과를 만든다."""
    return PageExtraction(
        document_type=DocumentType.LAB_RESULT, raw_text=raw_text, items=list(items)
    )


def fetch_document(session: Session, document_id: UUID) -> Row:
    """문서의 상태 컬럼을 읽는다."""
    return session.execute(
        text("SELECT status, status_changed_at, updated_at FROM documents WHERE id = :id"),
        {"id": document_id},
    ).one()


def fetch_analysis(session: Session, analysis_id: UUID) -> Row:
    """분석 작업의 상태와 실패 기록 컬럼을 읽는다."""
    return session.execute(
        text(
            "SELECT status, started_at, completed_at, model_version,"
            " error_code, failed_step, error_message, retryable"
            " FROM document_analyses WHERE id = :id"
        ),
        {"id": analysis_id},
    ).one()


def fetch_items(session: Session, analysis_id: UUID) -> Sequence[Row]:
    """분석의 추출 항목을 페이지 번호, 원문 순서로 읽는다. page_id는 페이지 번호로 바꿔 읽는다."""
    return session.execute(
        text(
            "SELECT p.page_no, i.item_type, i.source_text, i.confidence, i.source_box,"
            " i.source_box IS NULL AS box_is_sql_null"
            " FROM extracted_items i JOIN document_pages p ON p.id = i.page_id"
            " WHERE i.analysis_id = :id"
            " ORDER BY p.page_no, i.source_text"
        ),
        {"id": analysis_id},
    ).all()


def fetch_pages(session: Session, document_id: UUID) -> list[tuple[int, str, str | None]]:
    """문서의 페이지별 (page_no, ocr_status, ocr_text)를 페이지 번호 순서로 읽는다."""
    rows = session.execute(
        text(
            "SELECT page_no, ocr_status, ocr_text FROM document_pages"
            " WHERE document_id = :id ORDER BY page_no"
        ),
        {"id": document_id},
    ).all()
    return [tuple(row) for row in rows]


@pytest.fixture
def repository(session_factory: sessionmaker[Session]) -> SqlAlchemyDocumentAnalysisRepository:
    """테스트 연결에 묶인 저장소."""
    return SqlAlchemyDocumentAnalysisRepository(session_factory, stale_after=STALE_AFTER)


@pytest.fixture
def target(
    repository: SqlAlchemyDocumentAnalysisRepository, document_id: UUID, analysis_id: UUID
) -> AnalysisTarget:
    """2페이지가 저장된 문서와 PENDING 분석 작업."""
    repository.save_rendered_pages(
        document_id,
        [
            RenderedPage(
                page_no=page_no,
                width_px=100,
                height_px=200,
                storage_key=rendered_page_key(document_id, page_no),
            )
            for page_no in (1, 2)
        ],
        STARTED_AT,
    )
    found = repository.find_target(document_id)
    assert found is not None
    return found


def test_find_target_returns_document_and_latest_analysis(
    db_session: Session,
    repository: SqlAlchemyDocumentAnalysisRepository,
    document_id: UUID,
    analysis_id: UUID,
) -> None:
    """분석 작업이 여러 개면 created_at이 가장 늦은 것을 고르고, 문서의 키와 형식을 함께 준다."""
    newer_analysis_id = uuid4()
    db_session.execute(
        text(
            "INSERT INTO document_analyses (id, document_id, parser_version, created_at)"
            " VALUES (:id, :document_id, 'v1', CURRENT_TIMESTAMP + INTERVAL '1 second')"
        ),
        {"id": newer_analysis_id, "document_id": document_id},
    )

    assert repository.find_target(document_id) == AnalysisTarget(
        document_id=document_id,
        analysis_id=newer_analysis_id,
        storage_key=f"documents/{document_id}/original.pdf",
        mime_type="application/pdf",
    )


def test_find_target_without_document_or_analysis_is_none(
    repository: SqlAlchemyDocumentAnalysisRepository, document_id: UUID
) -> None:
    """문서가 없거나 분석 작업이 없으면 None이다."""
    assert repository.find_target(uuid4()) is None
    assert repository.find_target(document_id) is None


@pytest.mark.parametrize(
    "delete_sql",
    [
        "UPDATE documents SET status = 'DELETED' WHERE id = :id",
        "UPDATE documents SET deleted_at = CURRENT_TIMESTAMP WHERE id = :id",
    ],
)
def test_deleted_document_is_not_found_and_not_changed(
    db_session: Session,
    repository: SqlAlchemyDocumentAnalysisRepository,
    target: AnalysisTarget,
    delete_sql: str,
) -> None:
    """삭제된 문서는 대상으로 찾지 않고, 상태를 바꾸는 메서드도 문서·분석·페이지를 바꾸지 않는다."""
    db_session.execute(text(delete_sql), {"id": target.document_id})
    before = fetch_document(db_session, target.document_id)

    assert repository.find_target(target.document_id) is None
    assert repository.mark_started(target, STARTED_AT) is False
    assert (
        repository.save_result(
            target,
            DocumentStatus.READY,
            {1: make_extraction(make_item("a"))},
            "fake-model",
            COMPLETED_AT,
        )
        is False
    )
    assert repository.mark_failed(target, FAILURE, COMPLETED_AT) is False

    assert fetch_document(db_session, target.document_id) == before
    assert fetch_analysis(db_session, target.analysis_id).status == AnalysisStatus.PENDING
    assert fetch_items(db_session, target.analysis_id) == []
    assert fetch_pages(db_session, target.document_id) == UNTOUCHED_PAGES


def test_mark_started_sets_processing_and_running(
    db_session: Session, repository: SqlAlchemyDocumentAnalysisRepository, target: AnalysisTarget
) -> None:
    """시작하면 문서는 PROCESSING, 분석은 RUNNING이 되고 시각이 함께 바뀐다."""
    assert repository.mark_started(target, STARTED_AT) is True

    document = fetch_document(db_session, target.document_id)
    analysis = fetch_analysis(db_session, target.analysis_id)
    assert document.status == DocumentStatus.PROCESSING
    assert (document.status_changed_at, document.updated_at) == (STARTED_AT, STARTED_AT)
    assert (analysis.status, analysis.started_at) == (AnalysisStatus.RUNNING, STARTED_AT)
    assert analysis.completed_at is None


@pytest.mark.parametrize("document_status", [DocumentStatus.READY, DocumentStatus.NEEDS_REVIEW])
def test_save_result_stores_items_and_final_status(
    db_session: Session,
    repository: SqlAlchemyDocumentAnalysisRepository,
    target: AnalysisTarget,
    document_status: DocumentStatus,
) -> None:
    """항목과 페이지 원문을 저장하고, 분석은 SUCCEEDED, 문서는 받은 상태가 된다."""
    repository.mark_started(target, STARTED_AT)
    extractions = {
        1: make_extraction(make_item("a"), make_item("b", source_box=None), raw_text="1쪽 원문"),
        2: make_extraction(make_item("c"), raw_text="2쪽 원문"),
    }

    assert (
        repository.save_result(target, document_status, extractions, "fake-model", COMPLETED_AT)
        is True
    )

    rows = fetch_items(db_session, target.analysis_id)
    document = fetch_document(db_session, target.document_id)
    analysis = fetch_analysis(db_session, target.analysis_id)
    assert [(row.page_no, row.source_text) for row in rows] == [(1, "a"), (1, "b"), (2, "c")]
    assert rows[0].item_type == ExtractedItemType.LAB_VALUE
    assert rows[0].confidence == Decimal("0.9")
    assert rows[0].source_box == {"x": 10, "y": 20, "width": 30, "height": 40}
    assert [row.box_is_sql_null for row in rows] == [False, True, False]
    assert (document.status, document.status_changed_at) == (document_status, COMPLETED_AT)
    assert (analysis.status, analysis.completed_at) == (AnalysisStatus.SUCCEEDED, COMPLETED_AT)
    assert analysis.model_version == "fake-model"
    assert (analysis.error_code, analysis.failed_step, analysis.error_message) == (None, None, None)
    assert fetch_pages(db_session, target.document_id) == [
        (1, PageOcrStatus.SUCCEEDED, "1쪽 원문"),
        (2, PageOcrStatus.SUCCEEDED, "2쪽 원문"),
    ]


def test_page_missing_from_result_is_failed_without_text(
    db_session: Session, repository: SqlAlchemyDocumentAnalysisRepository, target: AnalysisTarget
) -> None:
    """결과에 없는(추출에 실패한) 페이지는 FAILED이고 원문이 없다."""
    extractions = {1: make_extraction(make_item("a"), raw_text="1쪽 원문")}

    repository.save_result(
        target, DocumentStatus.NEEDS_REVIEW, extractions, "fake-model", COMPLETED_AT
    )

    assert fetch_pages(db_session, target.document_id) == [
        (1, PageOcrStatus.SUCCEEDED, "1쪽 원문"),
        (2, PageOcrStatus.FAILED, None),
    ]


def test_blank_page_is_skipped_without_text(
    db_session: Session, repository: SqlAlchemyDocumentAnalysisRepository, target: AnalysisTarget
) -> None:
    """글자가 없는 페이지는 SKIPPED이고 원문은 빈 문자열이 아니라 NULL이다."""
    extractions = {1: make_extraction(make_item("a"), raw_text="1쪽 원문"), 2: BLANK_PAGE}

    repository.save_result(target, DocumentStatus.READY, extractions, "fake-model", COMPLETED_AT)

    assert fetch_pages(db_session, target.document_id) == [
        (1, PageOcrStatus.SUCCEEDED, "1쪽 원문"),
        (2, PageOcrStatus.SKIPPED, None),
    ]


def test_rerun_overwrites_page_text_and_status(
    db_session: Session, repository: SqlAlchemyDocumentAnalysisRepository, target: AnalysisTarget
) -> None:
    """다시 저장하면 원문은 새 값으로 바뀌고, 이번에 실패한 페이지는 예전 원문이 지워진다."""
    first = {
        1: make_extraction(make_item("a"), raw_text="처음 1쪽"),
        2: make_extraction(make_item("b"), raw_text="처음 2쪽"),
    }
    second = {1: make_extraction(make_item("c"), raw_text="다시 1쪽")}
    repository.save_result(target, DocumentStatus.READY, first, "fake-model", COMPLETED_AT)

    repository.save_result(target, DocumentStatus.NEEDS_REVIEW, second, "fake-model", COMPLETED_AT)

    assert fetch_pages(db_session, target.document_id) == [
        (1, PageOcrStatus.SUCCEEDED, "다시 1쪽"),
        (2, PageOcrStatus.FAILED, None),
    ]


def test_rerun_replaces_items_without_duplicates(
    db_session: Session, repository: SqlAlchemyDocumentAnalysisRepository, target: AnalysisTarget
) -> None:
    """같은 분석을 다시 저장하면 예전 항목은 사라지고 새 항목만 남는다."""
    first = {
        1: make_extraction(make_item("a"), make_item("b")),
        2: make_extraction(make_item("c")),
    }
    second = {1: make_extraction(make_item("d")), 2: make_extraction(make_item("e"))}
    repository.save_result(target, DocumentStatus.READY, first, "fake-model", COMPLETED_AT)

    repository.save_result(target, DocumentStatus.READY, second, "fake-model", COMPLETED_AT)

    rows = fetch_items(db_session, target.analysis_id)
    assert [(row.page_no, row.source_text) for row in rows] == [(1, "d"), (2, "e")]


def test_mark_failed_records_failure_and_removes_items(
    db_session: Session, repository: SqlAlchemyDocumentAnalysisRepository, target: AnalysisTarget
) -> None:
    """실패하면 문서·분석·페이지가 FAILED가 되고 실패 기록이 남으며, 예전 항목과 원문은 지워진다."""
    repository.save_result(
        target,
        DocumentStatus.READY,
        {1: make_extraction(make_item("a")), 2: make_extraction(make_item("b"))},
        "fake-model",
        COMPLETED_AT,
    )

    assert repository.mark_failed(target, FAILURE, COMPLETED_AT) is True

    document = fetch_document(db_session, target.document_id)
    analysis = fetch_analysis(db_session, target.analysis_id)
    assert (document.status, document.status_changed_at) == (DocumentStatus.FAILED, COMPLETED_AT)
    assert (analysis.status, analysis.completed_at) == (AnalysisStatus.FAILED, COMPLETED_AT)
    assert (analysis.error_code, analysis.failed_step) == ("VLM_TIMEOUT", "VLM_EXTRACTION")
    assert (analysis.error_message, analysis.retryable) == ("TIMEOUT", True)
    assert fetch_items(db_session, target.analysis_id) == []
    assert fetch_pages(db_session, target.document_id) == [
        (1, PageOcrStatus.FAILED, None),
        (2, PageOcrStatus.FAILED, None),
    ]


def test_succeeded_analysis_is_not_restarted(
    db_session: Session, repository: SqlAlchemyDocumentAnalysisRepository, target: AnalysisTarget
) -> None:
    """이미 SUCCEEDED인 분석은 다시 시작하지 않고 문서·분석·항목·페이지를 그대로 둔다."""
    repository.save_result(
        target,
        DocumentStatus.READY,
        {1: make_extraction(make_item("a"), raw_text="1쪽 원문")},
        "fake-model",
        COMPLETED_AT,
    )
    before = (
        fetch_document(db_session, target.document_id),
        fetch_analysis(db_session, target.analysis_id),
        fetch_items(db_session, target.analysis_id),
        fetch_pages(db_session, target.document_id),
    )

    assert repository.mark_started(target, STARTED_AT) is False

    assert (
        fetch_document(db_session, target.document_id),
        fetch_analysis(db_session, target.analysis_id),
        fetch_items(db_session, target.analysis_id),
        fetch_pages(db_session, target.document_id),
    ) == before


def test_running_analysis_before_stale_is_not_restarted(
    db_session: Session, repository: SqlAlchemyDocumentAnalysisRepository, target: AnalysisTarget
) -> None:
    """RUNNING이고 stale_after가 지나지 않았으면 다시 시작하지 않고 아무것도 바꾸지 않는다."""
    repository.mark_started(target, STARTED_AT)
    before = (
        fetch_document(db_session, target.document_id),
        fetch_analysis(db_session, target.analysis_id),
        fetch_pages(db_session, target.document_id),
    )

    assert repository.mark_started(target, STARTED_AT + STALE_AFTER - timedelta(minutes=1)) is False

    assert (
        fetch_document(db_session, target.document_id),
        fetch_analysis(db_session, target.analysis_id),
        fetch_pages(db_session, target.document_id),
    ) == before


def test_running_analysis_after_stale_is_restarted(
    db_session: Session, repository: SqlAlchemyDocumentAnalysisRepository, target: AnalysisTarget
) -> None:
    """RUNNING이어도 시작한 지 stale_after가 지나면 멈춘 것으로 보고 다시 시작한다."""
    repository.mark_started(target, STARTED_AT)
    restarted_at = STARTED_AT + STALE_AFTER

    assert repository.mark_started(target, restarted_at) is True

    analysis = fetch_analysis(db_session, target.analysis_id)
    assert (analysis.status, analysis.started_at) == (AnalysisStatus.RUNNING, restarted_at)


def test_running_analysis_without_started_at_is_restarted(
    db_session: Session, repository: SqlAlchemyDocumentAnalysisRepository, target: AnalysisTarget
) -> None:
    """시작 시각이 없는 RUNNING은 만료를 판단할 수 없어 다시 시작한다."""
    db_session.execute(
        text("UPDATE document_analyses SET status = 'RUNNING', started_at = NULL WHERE id = :id"),
        {"id": target.analysis_id},
    )

    assert repository.mark_started(target, STARTED_AT) is True


def test_mark_failed_keeps_items_referenced_by_explanation(
    db_session: Session,
    repository: SqlAlchemyDocumentAnalysisRepository,
    target: AnalysisTarget,
    explained_item_id: UUID,
) -> None:
    """설명이 참조하는 항목이 있어도 실패를 기록하고, 그 항목은 지우지 않고 남긴다."""
    repository.mark_started(target, STARTED_AT)

    assert repository.mark_failed(target, FAILURE, COMPLETED_AT) is True

    document = fetch_document(db_session, target.document_id)
    analysis = fetch_analysis(db_session, target.analysis_id)
    assert document.status == DocumentStatus.FAILED
    assert (analysis.status, analysis.error_code) == (AnalysisStatus.FAILED, "VLM_TIMEOUT")
    assert fetch_pages(db_session, target.document_id) == [
        (1, PageOcrStatus.FAILED, None),
        (2, PageOcrStatus.FAILED, None),
    ]
    remaining = db_session.scalars(
        text("SELECT id FROM extracted_items WHERE analysis_id = :id"),
        {"id": target.analysis_id},
    ).all()
    assert remaining == [explained_item_id]


def test_mark_failed_stores_page_split_failure(
    db_session: Session, repository: SqlAlchemyDocumentAnalysisRepository, target: AnalysisTarget
) -> None:
    """페이지 분리 실패의 오류 코드·단계·세부 코드가 그대로 저장된다."""
    failure = AnalysisFailure(
        error_code=AnalysisErrorCode.DOCUMENT_PAGE_SPLIT_FAILED,
        failed_step=FailedStep.PAGE_SPLIT,
        retryable=False,
        detail_code="PASSWORD_PROTECTED",
    )

    repository.mark_failed(target, failure, COMPLETED_AT)

    analysis = fetch_analysis(db_session, target.analysis_id)
    assert analysis.error_code == "DOCUMENT_PAGE_SPLIT_FAILED"
    assert analysis.failed_step == "PAGE_SPLIT"
    assert (analysis.error_message, analysis.retryable) == ("PASSWORD_PROTECTED", False)


def test_mark_started_clears_previous_failure(
    db_session: Session, repository: SqlAlchemyDocumentAnalysisRepository, target: AnalysisTarget
) -> None:
    """실패한 분석을 다시 시작하면 이전 실패 기록과 완료 시각이 비워진다."""
    repository.mark_failed(target, FAILURE, COMPLETED_AT)

    assert repository.mark_started(target, STARTED_AT) is True

    analysis = fetch_analysis(db_session, target.analysis_id)
    assert (analysis.status, analysis.completed_at) == (AnalysisStatus.RUNNING, None)
    assert (analysis.error_code, analysis.failed_step, analysis.error_message) == (None, None, None)
    assert analysis.retryable is False
