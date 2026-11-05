"""Spring 요청과 분리해 문서 하나의 분석을 조립부터 실행까지 돌리는 백그라운드 작업이다."""

import logging
import os
import traceback
from concurrent.futures import Executor
from datetime import UTC, datetime, timedelta
from uuid import UUID

from app.application.services.retrying_page_extractor import RetryingPageExtractor
from app.application.use_cases.analyze_document import analyze_document
from app.application.use_cases.index_document_chunks import (
    ChunkIndexOutcome,
    ChunkIndexStatus,
    index_document_chunks,
)
from app.core.config import Settings, get_settings
from app.domain.document.document_analysis import AnalysisFailure, DocumentStatus
from app.domain.operation.error_code import AnalysisErrorCode, FailedStep
from app.infrastructure.db.repositories.document_analysis_repository import (
    SqlAlchemyDocumentAnalysisRepository,
)
from app.infrastructure.db.repositories.document_chunk_repository import (
    SqlAlchemyDocumentChunkRepository,
)
from app.infrastructure.db.session import get_session_factory
from app.infrastructure.providers.embedding.openai_text_embedder import (
    OpenAITextEmbedder,
    create_text_embedder,
    has_embedding_api_key,
)
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
# 분석이 성공해 원문이 저장된 상태. 이 상태로 끝나면 같은 실행 안에서 청킹·임베딩까지 한다.
_INDEXABLE_STATUSES = frozenset({DocumentStatus.READY, DocumentStatus.NEEDS_REVIEW})


def run_document_analysis(
    document_id: UUID,
    request_id: str | None,
    object_key: str | None = None,
    mime_type: str | None = None,
) -> None:
    """문서 하나를 분석한다. 실패해도 예외를 밖으로 던지지 않고 로그만 남긴다.

    object_key·mime_type은 DB 값과 비교해 로그만 남기고, 분석에는 DB 값을 쓴다.
    서버 시작 때 다시 실행하는 경우에는 요청 값이 없어 모두 None이다.
    """
    repository: SqlAlchemyDocumentAnalysisRepository | None = None
    inner: OpenAIPageExtractor | None = None
    try:
        settings = get_settings()
        storage = LocalDocumentStorage(settings.storage_local_root)
        repository = _create_repository(settings)
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
        if result.document_status in _INDEXABLE_STATUSES:
            run_chunk_indexing(document_id, request_id)
    finally:
        if inner is not None:
            _close_extractor(inner, document_id)


def run_chunk_indexing(document_id: UUID, request_id: str | None = None) -> None:
    """문서 하나의 원문을 청킹·임베딩해 저장한다. 실패해도 예외를 던지지 않고 로그만 남긴다.

    분석 뒤의 별도 단계라 실패해도 문서 상태와 추출 항목은 바꾸지 않는다.
    """
    embedder: OpenAITextEmbedder | None = None
    try:
        embedder = create_text_embedder(get_settings())
        outcome = index_document_chunks(
            document_id,
            repository=SqlAlchemyDocumentChunkRepository(get_session_factory()),
            embedder=embedder,
        )
    except Exception as error:
        # 백그라운드 작업 경계다. 예외 메시지에는 원문이 섞일 수 있어 클래스 이름만 남긴다.
        logger.error(
            "조각 색인 작업 실패: document_id=%s, request_id=%s, error=%s",
            document_id,
            request_id,
            type(error).__name__,
        )
    else:
        _log_index_outcome(outcome, document_id, request_id)
    finally:
        if embedder is not None:
            _close_embedder(embedder, document_id)


def _log_index_outcome(
    outcome: ChunkIndexOutcome, document_id: UUID, request_id: str | None
) -> None:
    # 결과는 로그로만 남긴다. 문서 상태에 색인 결과를 기록하는 칼럼은 아직 없다.
    level = logging.ERROR if outcome.status == ChunkIndexStatus.FAILED else logging.INFO
    logger.log(
        level,
        "조각 색인 결과: document_id=%s, request_id=%s, status=%s, reason=%s, chunks=%d,"
        " succeeded_pages=%d, failed_pages=%d, error_code=%s",
        document_id,
        request_id,
        outcome.status,
        outcome.reason,
        outcome.chunk_count,
        outcome.succeeded_pages,
        outcome.failed_pages,
        outcome.error_code,
    )


def resume_unfinished_analyses(executor: Executor) -> None:
    """프로세스와 함께 멈춘 분석(오래된 PENDING, 만료된 RUNNING)을 executor에 다시 넣는다.

    이어서 분석은 끝났지만 조각 임베딩이 없는 문서의 색인도 넣는다.
    """
    try:
        settings = get_settings()
        document_ids = _create_repository(settings).find_resumable_document_ids(
            datetime.now(UTC),
            pending_after=timedelta(minutes=settings.document_analysis_resume_pending_minutes),
            limit=settings.document_analysis_resume_limit,
        )
        for document_id in document_ids:
            executor.submit(run_document_analysis, document_id, None)
    except Exception as error:
        # 서버 시작 때 executor에서 도는 작업이라 예외를 받을 곳이 없으므로 로그만 남긴다.
        logger.error("남은 문서 분석 재실행 실패: error=%s", type(error).__name__)
    else:
        logger.info("남은 문서 분석 재실행: count=%s", len(document_ids))
    _resume_unindexed_documents(executor)


def _resume_unindexed_documents(executor: Executor) -> None:
    # 분석 저장 직후 프로세스가 죽었거나 임베딩이 실패한 문서는 다른 재시도 경로가 없다.
    try:
        settings = get_settings()
        if not has_embedding_api_key(settings):
            # 키 없이 돌리면 문서마다 MISSING_API_KEY 오류만 쌓이므로 아예 넣지 않는다.
            logger.info("EMBEDDING_API_KEY가 없어 남은 조각 색인 재실행을 건너뜁니다")
            return
        document_ids = SqlAlchemyDocumentChunkRepository(
            get_session_factory()
        ).find_unindexed_document_ids(limit=settings.document_analysis_resume_limit)
        for document_id in document_ids:
            executor.submit(run_chunk_indexing, document_id, None)
    except Exception as error:
        logger.error("남은 조각 색인 재실행 실패: error=%s", type(error).__name__)
        return
    logger.info("남은 조각 색인 재실행: count=%s", len(document_ids))


def _create_repository(settings: Settings) -> SqlAlchemyDocumentAnalysisRepository:
    return SqlAlchemyDocumentAnalysisRepository(
        get_session_factory(),
        stale_after=timedelta(minutes=settings.document_analysis_stale_minutes),
    )


def _warn_if_request_differs(
    repository: SqlAlchemyDocumentAnalysisRepository,
    document_id: UUID,
    request_id: str | None,
    object_key: str | None,
    mime_type: str | None,
) -> None:
    if object_key is None or mime_type is None:
        return
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
    repository: SqlAlchemyDocumentAnalysisRepository, document_id: UUID, request_id: str | None
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


def _close_embedder(embedder: OpenAITextEmbedder, document_id: UUID) -> None:
    # 닫기 실패가 작업 밖으로 나가지 않게 여기서 막는다.
    try:
        embedder.close()
    except Exception as error:
        logger.error(
            "임베더 닫기 실패: document_id=%s, error=%s", document_id, type(error).__name__
        )


def _last_frame(error: BaseException) -> tuple[str, int | None]:
    """예외가 난 마지막 프레임의 파일 이름과 줄 번호. 메시지는 읽지 않는다."""
    frames = traceback.extract_tb(error.__traceback__)
    if not frames:
        return "unknown", None
    return os.path.basename(frames[-1].filename), frames[-1].lineno
