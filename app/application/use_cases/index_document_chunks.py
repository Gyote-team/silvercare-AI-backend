"""분석이 끝난 문서의 원문을 조각으로 나누고 임베딩해 검색용으로 저장한다."""

import logging
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from app.application.ports.document_chunk_repository import (
    ChunkIndexSkipReason,
    DocumentChunkRepository,
)
from app.application.ports.text_embedder import EmbeddingError, TextEmbedder
from app.pipelines.documents.steps.chunk_pages import chunk_pages

logger = logging.getLogger(__name__)


class ChunkIndexStatus(StrEnum):
    """색인 한 번의 결과."""

    INDEXED = "INDEXED"
    SKIPPED = "SKIPPED"  # 대상이 아니거나, 그사이 새 분석이 생겼거나 문서가 삭제됨
    FAILED = "FAILED"  # 임베딩 실패. 이전 조각과 임베딩은 그대로 남는다


@dataclass(frozen=True)
class ChunkIndexOutcome:
    """색인 결과. error_code는 FAILED일 때 EmbeddingErrorCode 값, reason은 SKIPPED일 때 이유다."""

    status: ChunkIndexStatus
    chunk_count: int = 0
    error_code: str | None = None
    reason: ChunkIndexSkipReason | None = None


def index_document_chunks(
    document_id: UUID, *, repository: DocumentChunkRepository, embedder: TextEmbedder
) -> ChunkIndexOutcome:
    """최신 성공 분석의 원문을 조각·임베딩해 이전 것과 바꾸고 결과를 반환한다.

    임베딩 실패는 FAILED로 돌려준다. DB 오류 같은 예상 밖 예외는 그대로 올린다.
    임베딩은 DB 트랜잭션 밖에서 먼저 끝내고, 교체는 저장소가 한 트랜잭션으로 한다.
    """
    target = repository.find_index_target(document_id)
    if target is None:
        logger.info("색인 대상이 아니어서 건너뜁니다: document_id=%s", document_id)
        return ChunkIndexOutcome(status=ChunkIndexStatus.SKIPPED)

    chunks = chunk_pages(target.pages)
    if not chunks:
        # 빈 결과로 교체하면 이전 분석의 색인까지 지워 검색할 것이 없어지므로 그대로 둔다.
        logger.info(
            "조각이 없어 기존 색인을 그대로 둡니다: document_id=%s, analysis_id=%s",
            document_id,
            target.analysis_id,
        )
        return ChunkIndexOutcome(
            status=ChunkIndexStatus.SKIPPED, reason=ChunkIndexSkipReason.NO_CHUNKS
        )
    try:
        vectors = embedder.embed([chunk.content for chunk in chunks])
    except EmbeddingError as error:
        logger.error(
            "조각 임베딩 실패: document_id=%s, analysis_id=%s, error_code=%s",
            document_id,
            target.analysis_id,
            error.code,
        )
        return ChunkIndexOutcome(status=ChunkIndexStatus.FAILED, error_code=error.code.value)

    if not repository.replace_chunks(target, chunks, vectors, embedder.model_name):
        return ChunkIndexOutcome(status=ChunkIndexStatus.SKIPPED)
    logger.info(
        "조각 색인 완료: document_id=%s, analysis_id=%s, chunks=%d",
        document_id,
        target.analysis_id,
        len(chunks),
    )
    return ChunkIndexOutcome(status=ChunkIndexStatus.INDEXED, chunk_count=len(chunks))
