"""분석이 끝난 문서의 원문을 조각으로 나누고 임베딩해 검색용으로 저장한다."""

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from app.application.ports.document_chunk_repository import (
    ChunkIndexSkipReason,
    ChunkIndexTarget,
    DocumentChunkRepository,
)
from app.application.ports.text_embedder import EmbeddingError, TextEmbedder
from app.pipelines.documents.steps.chunk_pages import chunk_pages


class ChunkIndexStatus(StrEnum):
    """색인 한 번의 결과."""

    INDEXED = "INDEXED"
    SKIPPED = "SKIPPED"  # 이유는 ChunkIndexOutcome.reason에 담는다
    FAILED = "FAILED"  # 임베딩 실패. 이전 조각과 임베딩은 그대로 남는다


@dataclass(frozen=True)
class ChunkIndexOutcome:
    """색인 결과. 로그는 호출한 쪽이 이 값으로 남긴다.

    reason은 SKIPPED일 때 이유, error_code는 FAILED일 때 EmbeddingErrorCode 값이다.
    succeeded_pages는 원문이 있는 SUCCEEDED 페이지 수, failed_pages는 FAILED 페이지 수다.
    """

    status: ChunkIndexStatus
    chunk_count: int = 0
    error_code: str | None = None
    reason: ChunkIndexSkipReason | None = None
    succeeded_pages: int = 0
    failed_pages: int = 0


def index_document_chunks(
    document_id: UUID, *, repository: DocumentChunkRepository, embedder: TextEmbedder
) -> ChunkIndexOutcome:
    """최신 성공 분석의 원문을 조각·임베딩해 이전 것과 바꾸고 결과를 반환한다.

    임베딩 실패는 FAILED로 돌려준다. DB 오류 같은 예상 밖 예외는 그대로 올린다.
    임베딩은 DB 트랜잭션 밖에서 먼저 끝내고, 교체는 저장소가 한 트랜잭션으로 한다.
    """
    target = repository.find_index_target(document_id)
    if isinstance(target, ChunkIndexSkipReason):
        return _skipped(target)
    if not target.pages:
        return _skipped(ChunkIndexSkipReason.NO_SOURCE_TEXT, target)

    chunks = chunk_pages(target.pages)
    if not chunks:
        # 빈 결과로 교체하면 이전 분석의 색인까지 지워 검색할 것이 없어지므로 그대로 둔다.
        return _skipped(ChunkIndexSkipReason.NO_CHUNKS, target)
    try:
        vectors = embedder.embed([chunk.content for chunk in chunks])
    except EmbeddingError as error:
        return ChunkIndexOutcome(
            status=ChunkIndexStatus.FAILED,
            error_code=error.code.value,
            **_page_counts(target),
        )

    reason = repository.replace_chunks(target, chunks, vectors, embedder.model_name)
    if reason is not None:
        return _skipped(reason, target)
    return ChunkIndexOutcome(
        status=ChunkIndexStatus.INDEXED, chunk_count=len(chunks), **_page_counts(target)
    )


def _skipped(
    reason: ChunkIndexSkipReason, target: ChunkIndexTarget | None = None
) -> ChunkIndexOutcome:
    counts = _page_counts(target) if target is not None else {}
    return ChunkIndexOutcome(status=ChunkIndexStatus.SKIPPED, reason=reason, **counts)


def _page_counts(target: ChunkIndexTarget) -> dict[str, int]:
    return {"succeeded_pages": len(target.pages), "failed_pages": target.failed_page_count}
