"""documentId 하나를 분석하는 유스케이스.

원본 읽기 → 페이지 저장 → VLM 추출 → 추출 결과와 최종 상태 기록 순서로 실행한다.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from app.application.ports.document_analysis_repository import (
    AnalysisTarget,
    DocumentAnalysisRepository,
)
from app.application.ports.document_page_repository import DocumentPageRepository
from app.application.ports.document_storage import (
    DocumentNotFoundError,
    InvalidObjectKeyError,
    OriginalDocumentReader,
    RenderedPageWriteError,
    RenderedPageWriter,
)
from app.application.ports.page_extractor import (
    PageExtractionErrorCode,
    PageExtractor,
    VlmUsage,
)
from app.domain.document.document_analysis import (
    AnalysisErrorCode,
    AnalysisFailure,
    DocumentStatus,
    FailedStep,
)
from app.domain.document.page_split import PageSplitError
from app.pipelines.documents.steps.extract_pages import PageExtractionOutcome, extract_pages
from app.pipelines.documents.steps.load_document_pages import load_document_pages
from app.pipelines.documents.steps.save_rendered_pages import save_rendered_pages

logger = logging.getLogger(__name__)

_UNEXPECTED_DETAIL_CODE = "UNEXPECTED_ERROR"

# 저장소 예외 → (error_message에 넣는 세부 코드, retryable).
# FC-15에 저장 실패 전용 값이 없어 모두 OBJECT_FETCH_FAILED / OBJECT_FETCH로 기록한다.
_STORAGE_FAILURES: dict[type[Exception], tuple[str, bool]] = {
    DocumentNotFoundError: ("DOCUMENT_NOT_FOUND", True),
    InvalidObjectKeyError: ("INVALID_OBJECT_KEY", False),
    RenderedPageWriteError: ("RENDERED_PAGE_WRITE_FAILED", True),
}

# VLM 오류 코드 → (error_code, failed_step, retryable).
_EXTRACTION_FAILURES: dict[
    PageExtractionErrorCode, tuple[AnalysisErrorCode, FailedStep, bool]
] = {
    PageExtractionErrorCode.MISSING_API_KEY: (
        AnalysisErrorCode.AI_GENERATION_FAILED,
        FailedStep.VLM_EXTRACTION,
        False,
    ),
    PageExtractionErrorCode.TIMEOUT: (
        AnalysisErrorCode.VLM_TIMEOUT,
        FailedStep.VLM_EXTRACTION,
        True,
    ),
    PageExtractionErrorCode.RATE_LIMITED: (
        AnalysisErrorCode.VLM_RATE_LIMIT,
        FailedStep.VLM_EXTRACTION,
        True,
    ),
    PageExtractionErrorCode.API_ERROR: (
        AnalysisErrorCode.AI_GENERATION_FAILED,
        FailedStep.VLM_EXTRACTION,
        True,
    ),
    PageExtractionErrorCode.REQUEST_REJECTED: (
        AnalysisErrorCode.AI_GENERATION_FAILED,
        FailedStep.VLM_EXTRACTION,
        False,
    ),
    PageExtractionErrorCode.INCOMPLETE: (
        AnalysisErrorCode.AI_GENERATION_FAILED,
        FailedStep.VLM_EXTRACTION,
        False,
    ),
    PageExtractionErrorCode.REFUSED: (
        AnalysisErrorCode.AI_GENERATION_FAILED,
        FailedStep.VLM_EXTRACTION,
        False,
    ),
    PageExtractionErrorCode.SCHEMA_MISMATCH: (
        AnalysisErrorCode.STRUCTURE_VALIDATION_FAILED,
        FailedStep.STRUCTURE_VALIDATION,
        True,
    ),
}


@dataclass(frozen=True)
class DocumentAnalysisResult:
    """분석 한 번의 결과.

    document_status는 기록한 문서 상태이고, 아무것도 바꾸지 않았으면 None이다.
    usages는 성공한 VLM 호출의 사용량이다. 아직 DB에 기록하지 않고 담아 두기만 한다.
    """

    document_status: DocumentStatus | None
    failure: AnalysisFailure | None = None
    usages: tuple[VlmUsage, ...] = ()


def analyze_document(
    document_id: UUID,
    *,
    reader: OriginalDocumentReader,
    writer: RenderedPageWriter,
    repository: DocumentAnalysisRepository,
    page_repository: DocumentPageRepository,
    extractor: PageExtractor,
    render_dpi: int,
    max_pages: int,
) -> DocumentAnalysisResult:
    """문서 하나를 분석하고 결과와 상태를 기록한다.

    문서가 없거나 삭제됐거나 분석 작업이 없으면 아무것도 바꾸지 않는다.
    예상 밖 예외가 나면 FAILED로 기록한 뒤 그 예외를 다시 올린다.
    """
    target = repository.find_target(document_id)
    # 시작 표시는 따로 commit해서 Spring 화면에 바로 보이게 한다.
    if target is None or not repository.mark_started(target, datetime.now(UTC)):
        logger.info("분석 대상이 아니어서 건너뜁니다: document_id=%s", document_id)
        return DocumentAnalysisResult(document_status=None)

    step = FailedStep.OBJECT_FETCH
    try:
        try:
            pages = load_document_pages(
                reader,
                target.storage_key,
                target.mime_type,
                render_dpi=render_dpi,
                max_pages=max_pages,
            )
            save_rendered_pages(writer, page_repository, document_id, pages)
        except (PageSplitError, *_STORAGE_FAILURES) as error:
            return _record_failure(repository, target, _fetch_failure(error))
        step = FailedStep.VLM_EXTRACTION
        # VLM 호출은 오래 걸리므로 DB 트랜잭션 밖에서 한다.
        outcomes = extract_pages(extractor, pages)
        return _record_outcomes(repository, target, len(pages), outcomes)
    except Exception as error:
        # 예외 메시지에는 원문이나 경로가 들어 있을 수 있어 클래스 이름만 남긴다.
        logger.error(
            "분석 중 예상 밖 오류: document_id=%s, analysis_id=%s, step=%s, error=%s",
            document_id,
            target.analysis_id,
            step,
            type(error).__name__,
        )
        _record_unexpected_failure(repository, target, step)
        raise


def _fetch_failure(error: Exception) -> AnalysisFailure:
    if isinstance(error, PageSplitError):
        detail_code, retryable = error.code.value, False
    else:
        detail_code, retryable = _STORAGE_FAILURES[type(error)]
    return AnalysisFailure(
        error_code=AnalysisErrorCode.OBJECT_FETCH_FAILED,
        failed_step=FailedStep.OBJECT_FETCH,
        retryable=retryable,
        detail_code=detail_code,
    )


def _record_outcomes(
    repository: DocumentAnalysisRepository,
    target: AnalysisTarget,
    page_count: int,
    outcomes: Sequence[PageExtractionOutcome],
) -> DocumentAnalysisResult:
    succeeded = {
        outcome.page_no: outcome.result for outcome in outcomes if outcome.result is not None
    }
    if not succeeded:
        # 페이지마다 실패 이유가 다르면 첫 실패 페이지의 이유를 기록한다.
        code = next(outcome.error_code for outcome in outcomes if outcome.error_code is not None)
        error_code, failed_step, retryable = _EXTRACTION_FAILURES[code]
        failure = AnalysisFailure(
            error_code=error_code,
            failed_step=failed_step,
            retryable=retryable,
            detail_code=code.value,
        )
        return _record_failure(repository, target, failure)

    usages = tuple(result.usage for result in succeeded.values())
    status = DocumentStatus.READY if len(succeeded) == page_count else DocumentStatus.NEEDS_REVIEW
    saved = repository.save_result(
        target,
        status,
        {page_no: result.extraction for page_no, result in succeeded.items()},
        usages[0].model_name,
        datetime.now(UTC),
    )
    logger.info(
        "분석 완료: document_id=%s, analysis_id=%s, status=%s, pages=%d/%d",
        target.document_id,
        target.analysis_id,
        status if saved else None,
        len(succeeded),
        page_count,
    )
    return DocumentAnalysisResult(document_status=status if saved else None, usages=usages)


def _record_failure(
    repository: DocumentAnalysisRepository, target: AnalysisTarget, failure: AnalysisFailure
) -> DocumentAnalysisResult:
    saved = repository.mark_failed(target, failure, datetime.now(UTC))
    logger.warning(
        "분석 실패: document_id=%s, analysis_id=%s, error_code=%s, failed_step=%s, detail=%s",
        target.document_id,
        target.analysis_id,
        failure.error_code,
        failure.failed_step,
        failure.detail_code,
    )
    return DocumentAnalysisResult(
        document_status=DocumentStatus.FAILED if saved else None, failure=failure
    )


def _record_unexpected_failure(
    repository: DocumentAnalysisRepository, target: AnalysisTarget, step: FailedStep
) -> None:
    failure = AnalysisFailure(
        error_code=AnalysisErrorCode.AI_GENERATION_FAILED,
        failed_step=step,
        retryable=True,
        detail_code=_UNEXPECTED_DETAIL_CODE,
    )
    # 기록까지 실패해도 원래 예외를 가리지 않게 로그만 남긴다.
    try:
        repository.mark_failed(target, failure, datetime.now(UTC))
    except Exception as error:
        logger.error(
            "실패 기록 중 오류: document_id=%s, analysis_id=%s, error=%s",
            target.document_id,
            target.analysis_id,
            type(error).__name__,
        )
