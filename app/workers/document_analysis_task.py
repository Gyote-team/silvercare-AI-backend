"""Spring 요청과 분리해 문서 하나의 분석을 조립부터 실행까지 돌리는 백그라운드 작업이다."""

import logging
from uuid import UUID

from app.application.services.retrying_page_extractor import RetryingPageExtractor
from app.application.use_cases.analyze_document import analyze_document
from app.core.config import get_settings
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


def run_document_analysis(document_id: UUID, request_id: str) -> None:
    """문서 하나를 분석한다. 실패해도 예외를 밖으로 던지지 않고 로그만 남긴다."""
    inner: OpenAIPageExtractor | None = None
    try:
        settings = get_settings()
        storage = LocalDocumentStorage(settings.storage_local_root)
        repository = SqlAlchemyDocumentAnalysisRepository(get_session_factory())
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
        # 조립 단계(설정·세션·추출기)에서 실패하면 mark_started 전이라 문서가 PENDING에 남는다.
        # 예외 메시지에는 경로나 원문이 들어 있을 수 있어 클래스 이름만 남긴다.
        logger.error(
            "문서 분석 작업 실패: document_id=%s, request_id=%s, error=%s",
            document_id,
            request_id,
            type(error).__name__,
        )
    else:
        # document_status가 None이면 대상이 없거나 mark_started가 시작을 거부한 경우다.
        logger.info(
            "문서 분석 작업 종료: document_id=%s, request_id=%s, document_status=%s",
            document_id,
            request_id,
            result.document_status or "건너뜀(대상 없음 또는 시작 거부)",
        )
    finally:
        if inner is not None:
            inner.close()
