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
from app.pipelines.documents.steps.chunk_pages import chunk_pages, is_personal_info_line


class ChunkIndexStatus(StrEnum):
    """색인 한 번의 결과."""

    INDEXED = "INDEXED"
    SKIPPED = "SKIPPED"  # 이유는 ChunkIndexOutcome.reason에 담는다
    FAILED = "FAILED"  # 임베딩 실패. 이전 조각과 임베딩은 그대로 남는다


@dataclass(frozen=True)
class ChunkIndexOutcome:
    """색인 결과. 로그는 호출한 쪽이 이 값으로 남긴다.

    reason은 SKIPPED일 때 이유, error_code는 FAILED일 때 EmbeddingErrorCode 값이다.
    chunk_count는 최신 분석의 조각 수, embedded_count는 이번에 새로 임베딩한 조각 수다.
    succeeded_pages는 원문이 있는 SUCCEEDED 페이지 수, failed_pages는 FAILED 페이지 수다.
    """

    status: ChunkIndexStatus
    chunk_count: int = 0
    embedded_count: int = 0
    error_code: str | None = None
    reason: ChunkIndexSkipReason | None = None
    succeeded_pages: int = 0
    failed_pages: int = 0


def index_document_chunks(
    document_id: UUID, *, repository: DocumentChunkRepository, embedder: TextEmbedder
) -> ChunkIndexOutcome:
    """최신 성공 분석의 원문을 조각·임베딩해 이전 것과 바꾸고 결과를 반환한다.

    그 분석의 조각이 이미 있으면 다시 나누지 않고 임베딩이 없는 조각만 채운다.

    임베딩 실패는 FAILED로 돌려준다. DB 오류 같은 예상 밖 예외는 그대로 올린다.
    임베딩은 DB 트랜잭션 밖에서 먼저 끝내고, 교체는 저장소가 한 트랜잭션으로 한다.
    """
    target = repository.find_index_target(document_id, embedder.model_name)
    if isinstance(target, ChunkIndexSkipReason):
        return _skipped(target)
    if target.stored_chunk_count:
        return _fill_missing(target, repository=repository, embedder=embedder)
    if not target.pages:
        return _skipped(ChunkIndexSkipReason.NO_SOURCE_TEXT, target)

    # 환자 이름·연락처 같은 줄은 임베딩·검색 대상에 넣지 않는다.
    chunks = chunk_pages(target.pages, exclude_line=is_personal_info_line)
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
        status=ChunkIndexStatus.INDEXED,
        chunk_count=len(chunks),
        embedded_count=len(chunks),
        **_page_counts(target),
    )


def _fill_missing(
    target: ChunkIndexTarget, *, repository: DocumentChunkRepository, embedder: TextEmbedder
) -> ChunkIndexOutcome:
    # 이 분석의 조각이 이미 있으면 다시 나누지 않고, 임베딩이 없는 조각만 임베딩해 채운다.
    missing = target.missing_chunks
    if missing:
        try:
            vectors = embedder.embed([chunk.content for chunk in missing])
        except EmbeddingError as error:
            return ChunkIndexOutcome(
                status=ChunkIndexStatus.FAILED,
                chunk_count=target.stored_chunk_count,
                error_code=error.code.value,
                **_page_counts(target),
            )
        reason = repository.fill_missing_embeddings(
            target, [chunk.chunk_id for chunk in missing], vectors, embedder.model_name
        )
        if reason is not None:
            return _skipped(reason, target)
    return ChunkIndexOutcome(
        status=ChunkIndexStatus.INDEXED,
        chunk_count=target.stored_chunk_count,
        embedded_count=len(missing),
        **_page_counts(target),
    )


def _skipped(
    reason: ChunkIndexSkipReason, target: ChunkIndexTarget | None = None
) -> ChunkIndexOutcome:
    counts = _page_counts(target) if target is not None else {}
    return ChunkIndexOutcome(status=ChunkIndexStatus.SKIPPED, reason=reason, **counts)


def _page_counts(target: ChunkIndexTarget) -> dict[str, int]:
    return {"succeeded_pages": len(target.pages), "failed_pages": target.failed_page_count}
