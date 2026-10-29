"""Spring 요청과 분리해 문서 하나의 분석을 조립부터 실행까지 돌리는 백그라운드 작업이다."""

import logging
import os
import traceback
from datetime import UTC, datetime, timedelta
from uuid import UUID

from app.application.services.retrying_page_extractor import RetryingPageExtractor
from app.application.use_cases.analyze_document import analyze_document
from app.core.config import get_settings
from app.domain.document.document_analysis import AnalysisFailure
from app.domain.operation.error_code import AnalysisErrorCode, FailedStep
from app.infrastructure.db.repositories.document_analysis_repository import (
    SqlAlchemyDocumentAnalysisRepository,
)
from app.infrastructure.db.session import get_session_factory
from app.infrastructure.providers.llm.openai_page_extractor import (
    OpenAIPageExtractor,
    create_page_extractor,
)
from app.infrastructure.storage.local_document_storage import LocalDocumentStorage

logger = logging.getLogger(__name__)

# 분석을 시작하기 전(조립·대상 조회·시작 표시)에 실패했을 때 남기는 기록.
# FC-15에 조립 단계 값이 없어 analyze_document의 첫 단계(OBJECT_FETCH)로 기록한다.
_SETUP_FAILURE = AnalysisFailure(
    error_code=AnalysisErrorCode.AI_GENERATION_FAILED,
    failed_step=FailedStep.OBJECT_FETCH,
    retryable=True,
    detail_code="SETUP_FAILED",
)


def run_document_analysis(
    document_id: UUID, request_id: str, object_key: str, mime_type: str
) -> None:
    """문서 하나를 분석한다. 실패해도 예외를 밖으로 던지지 않고 로그만 남긴다.

    object_key·mime_type은 DB 값과 비교해 로그만 남기고, 분석에는 DB 값을 쓴다.
    """
    repository: SqlAlchemyDocumentAnalysisRepository | None = None
    inner: OpenAIPageExtractor | None = None
    try:
        settings = get_settings()
        storage = LocalDocumentStorage(settings.storage_local_root)
        repository = SqlAlchemyDocumentAnalysisRepository(
            get_session_factory(),
            stale_after=timedelta(minutes=settings.document_analysis_stale_minutes),
        )
        _warn_if_request_differs(repository, document_id, request_id, object_key, mime_type)
        inner = create_page_extractor(settings)
        extractor = RetryingPageExtractor(
            inner,
            max_retries=settings.openai_vlm_max_retries,
            backoff_seconds=settings.openai_vlm_retry_backoff_seconds,
        )
        result = analyze_document(
            document_id,
            reader=storage,
            writer=storage,
            repository=repository,
            page_repository=repository,
            extractor=extractor,
            render_dpi=settings.document_render_dpi,
            max_pages=settings.document_max_pages,
            max_page_side_px=settings.document_max_page_side_px,
        )
    except Exception as error:
        # 백그라운드 작업 경계라 예외를 올려도 받을 곳이 없으므로 로그만 남긴다.
        # 예외 메시지에는 경로나 원문이 들어 있을 수 있어 클래스 이름과 위치만 남긴다.
        file_name, line_no = _last_frame(error)
        logger.error(
            "문서 분석 작업 실패: document_id=%s, request_id=%s, error=%s, at=%s:%s",
            document_id,
            request_id,
            type(error).__name__,
            file_name,
            line_no,
        )
        if repository is not None:
            _record_setup_failure(repository, document_id, request_id)
    else:
        # document_status가 None이면 대상이 없거나, 시작을 거부했거나, 분석 도중 삭제된 경우다.
        logger.info(
            "문서 분석 작업 종료: document_id=%s, request_id=%s, document_status=%s",
            document_id,
            request_id,
            result.document_status or "건너뜀(대상 없음, 시작 거부 또는 분석 도중 삭제)",
        )
    finally:
        if inner is not None:
            _close_extractor(inner, document_id)


def _warn_if_request_differs(
    repository: SqlAlchemyDocumentAnalysisRepository,
    document_id: UUID,
    request_id: str,
    object_key: str,
    mime_type: str,
) -> None:
    target = repository.find_target(document_id)
    if target is None:
        return
    differs = [
        name
        for name, requested, stored in (
            ("objectKey", object_key, target.storage_key),
            ("mimeType", mime_type, target.mime_type),
        )
        if requested != stored
    ]
    if differs:
        # 값에는 경로가 들어 있으므로 필드 이름만 남긴다. 분석은 DB 값으로 계속한다.
        logger.warning(
            "요청 값이 DB와 다릅니다: document_id=%s, request_id=%s, fields=%s",
            document_id,
            request_id,
            ",".join(differs),
        )


def _record_setup_failure(
    repository: SqlAlchemyDocumentAnalysisRepository, document_id: UUID, request_id: str
) -> None:
    # 분석이 이미 RUNNING·FAILED로 바뀌었으면 PENDING 조건 때문에 아무것도 바꾸지 않는다.
    # 그래서 analyze_document가 실패를 기록한 뒤 다시 올린 예외에도 그대로 불러도 된다.
    try:
        repository.mark_failed_before_start(document_id, _SETUP_FAILURE, datetime.now(UTC))
    except Exception as error:
        # DB 연결 실패 등으로 기록할 수 없으면 로그만 남긴다.
        logger.error(
            "시작 전 실패 기록 중 오류: document_id=%s, request_id=%s, error=%s",
            document_id,
            request_id,
            type(error).__name__,
        )


def _close_extractor(inner: OpenAIPageExtractor, document_id: UUID) -> None:
    # 닫기 실패가 작업 밖으로 나가면 서버가 예외 메시지까지 로그에 남기므로 여기서 막는다.
    try:
        inner.close()
    except Exception as error:
        logger.error(
            "추출기 닫기 실패: document_id=%s, error=%s", document_id, type(error).__name__
        )


def _last_frame(error: BaseException) -> tuple[str, int | None]:
    """예외가 난 마지막 프레임의 파일 이름과 줄 번호. 메시지는 읽지 않는다."""
    frames = traceback.extract_tb(error.__traceback__)
    if not frames:
        return "unknown", None
    return os.path.basename(frames[-1].filename), frames[-1].lineno
