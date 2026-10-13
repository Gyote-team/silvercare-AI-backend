"""문서 하나로 analyze_document를 실제 DB, 로컬 저장소, OpenAI에 한 번 돌려 보는 수동 확인 스크립트.

자동 테스트에 포함되지 않는다. 실행할 때마다 OpenAI 비용이 들고 DB 상태가 바뀐다.
저장소 루트에서 실행한다. documentId를 주지 않으면 삭제되지 않은 가장 최근 문서를 고른다.

    python -m scripts.try_analyze_document [documentId]
"""

import sys
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.application.services.retrying_page_extractor import RetryingPageExtractor
from app.application.use_cases.analyze_document import analyze_document
from app.core.config import get_settings
from app.infrastructure.db.models.document import DocumentModel
from app.infrastructure.db.models.document_page import DocumentPageModel
from app.infrastructure.db.repositories.document_analysis_repository import (
    SqlAlchemyDocumentAnalysisRepository,
)
from app.infrastructure.db.session import get_session_factory
from app.infrastructure.providers.llm.openai_page_extractor import create_page_extractor
from app.infrastructure.storage.local_document_storage import LocalDocumentStorage


def pick_document_id(session_factory: sessionmaker[Session]) -> UUID:
    """인자로 받은 documentId를 쓰고, 없으면 삭제되지 않은 가장 최근 문서를 고른다."""
    if len(sys.argv) > 1:
        try:
            return UUID(sys.argv[1])
        except ValueError:
            print("documentId는 UUID 형식이어야 합니다.")
            raise SystemExit(1) from None
    with session_factory() as session:
        # documents 매핑에 created_at이 없어 updated_at이 가장 늦은 문서를 고른다.
        document_id = session.execute(
            select(DocumentModel.id)
            .where(DocumentModel.deleted_at.is_(None))
            .order_by(DocumentModel.updated_at.desc())
            .limit(1)
        ).scalar_one_or_none()
    if document_id is None:
        print("분석할 문서가 없습니다.")
        raise SystemExit(1)
    return document_id


def main() -> None:
    """문서 하나를 한 번 분석하고 문서 상태, 실패 기록, 페이지 수를 출력한다."""
    settings = get_settings()
    session_factory = get_session_factory()
    document_id = pick_document_id(session_factory)
    storage = LocalDocumentStorage(settings.storage_local_root)
    repository = SqlAlchemyDocumentAnalysisRepository(session_factory)
    inner = create_page_extractor(settings)
    extractor = RetryingPageExtractor(
        inner,
        max_retries=settings.openai_vlm_max_retries,
        backoff_seconds=settings.openai_vlm_retry_backoff_seconds,
    )

    print(f"document_id: {document_id}")
    try:
        try:
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
            # 예외 메시지에는 원문이나 경로가 들어 있을 수 있어 클래스 이름만 출력한다.
            print(f"예상 밖 오류: {type(error).__name__}")
            raise SystemExit(1) from None

        with session_factory() as session:
            saved_pages = session.execute(
                select(func.count())
                .select_from(DocumentPageModel)
                .where(DocumentPageModel.document_id == document_id)
            ).scalar_one()
    finally:
        inner.close()

    # objectKey, 경로, 추출 원문은 출력하지 않는다.
    print(f"document_status: {result.document_status or '변경 없음'}")
    if result.failure is not None:
        failure = result.failure
        print(
            f"failure: {failure.error_code} / {failure.failed_step} / {failure.detail_code}"
            f" (retryable={failure.retryable})"
        )
    print(f"페이지: 저장 {saved_pages}장, 추출 성공 {len(result.usages)}장")


if __name__ == "__main__":
    main()
