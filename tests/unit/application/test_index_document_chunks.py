"""조각 색인 유스케이스 테스트. 저장소와 임베더는 가짜로 바꾼다."""

from collections.abc import Sequence
from dataclasses import replace
from uuid import UUID, uuid4

import pytest

from app.application.ports.document_chunk_repository import (
    ChunkIndexSkipReason,
    ChunkIndexTarget,
    StoredChunk,
)
from app.application.ports.text_embedder import (
    EMBEDDING_DIMENSIONS,
    EmbeddingError,
    EmbeddingErrorCode,
)
from app.application.use_cases.index_document_chunks import (
    ChunkIndexOutcome,
    ChunkIndexStatus,
    index_document_chunks,
)
from app.domain.document.document_chunk import DocumentChunk, PageText

SOURCE_TEXT = "합성 검사 결과지\n혈색소 13.2 g/dL"


class FakeRepository:
    """대상 조회와 교체 호출만 흉내 내는 가짜 조각 저장소."""

    def __init__(
        self,
        target: ChunkIndexTarget | ChunkIndexSkipReason,
        *,
        replace_result: ChunkIndexSkipReason | None = None,
    ) -> None:
        self.target = target
        self.replace_result = replace_result
        self.replace_calls: list[tuple[list[DocumentChunk], list[list[float]], str]] = []
        self.fill_calls: list[tuple[list[UUID], int, str]] = []

    def find_index_target(
        self, document_id: UUID, model_name: str
    ) -> ChunkIndexTarget | ChunkIndexSkipReason:
        """정해 둔 대상이나 건너뛸 이유를 돌려준다."""
        return self.target

    def fill_missing_embeddings(
        self,
        target: ChunkIndexTarget,
        chunk_ids: Sequence[UUID],
        vectors: Sequence[Sequence[float]],
        model_name: str,
    ) -> ChunkIndexSkipReason | None:
        """받은 조각 id와 벡터 수를 기록하고 정해 둔 결과를 돌려준다."""
        self.fill_calls.append((list(chunk_ids), len(vectors), model_name))
        return self.replace_result

    def replace_chunks(
        self,
        target: ChunkIndexTarget,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
        model_name: str,
    ) -> ChunkIndexSkipReason | None:
        """받은 값을 기록하고 정해 둔 결과를 돌려준다."""
        self.replace_calls.append((list(chunks), [list(v) for v in vectors], model_name))
        return self.replace_result

    def find_unindexed_document_ids(self, *, model_name: str, limit: int) -> list[UUID]:
        """쓰지 않는다."""
        return []


class FakeEmbedder:
    """입력마다 같은 벡터를 돌려주는 가짜 임베더. 정해 둔 오류가 있으면 던진다."""

    model_name = "fake-embedding"

    def __init__(self, error: EmbeddingErrorCode | None = None) -> None:
        self.error = error
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """호출을 기록하고 입력 수만큼 벡터를 만든다."""
        self.calls.append(list(texts))
        if self.error is not None:
            raise EmbeddingError(self.error)
        return [[0.5] * EMBEDDING_DIMENSIONS for _ in texts]


def make_target(*texts: str, failed_page_count: int = 0) -> ChunkIndexTarget:
    """페이지 원문이 주어진 합성 색인 대상을 만든다."""
    return ChunkIndexTarget(
        document_id=uuid4(),
        analysis_id=uuid4(),
        patient_id=uuid4(),
        visit_id=uuid4(),
        pages=tuple(
            PageText(page_id=uuid4(), page_no=index + 1, text=text)
            for index, text in enumerate(texts)
        ),
        failed_page_count=failed_page_count,
    )


def run(repository: FakeRepository, embedder: FakeEmbedder) -> ChunkIndexOutcome:
    """가짜로 유스케이스를 한 번 실행한다."""
    return index_document_chunks(uuid4(), repository=repository, embedder=embedder)


def test_chunks_are_embedded_and_replaced() -> None:
    """조각을 임베딩하고 모델 이름과 함께 교체를 한 번 부르며, 페이지 수를 함께 돌려준다."""
    repository = FakeRepository(make_target(SOURCE_TEXT, "둘째 쪽", failed_page_count=1))
    embedder = FakeEmbedder()

    outcome = run(repository, embedder)

    assert outcome == ChunkIndexOutcome(
        status=ChunkIndexStatus.INDEXED,
        chunk_count=2,
        embedded_count=2,
        succeeded_pages=2,
        failed_pages=1,
    )
    assert embedder.calls == [[SOURCE_TEXT, "둘째 쪽"]]
    chunks, vectors, model_name = repository.replace_calls[0]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert len(vectors) == 2
    assert model_name == "fake-embedding"


@pytest.mark.parametrize(
    "reason", [ChunkIndexSkipReason.DELETED, ChunkIndexSkipReason.NOT_ANALYZED]
)
def test_lookup_reason_skips_without_embedding(reason: ChunkIndexSkipReason) -> None:
    """대상 조회가 이유를 돌려주면 임베딩도 교체도 하지 않고 그 이유로 건너뛴다."""
    repository = FakeRepository(reason)
    embedder = FakeEmbedder()

    assert run(repository, embedder) == ChunkIndexOutcome(
        status=ChunkIndexStatus.SKIPPED, reason=reason
    )
    assert embedder.calls == []
    assert repository.replace_calls == []


def test_no_text_pages_skip_as_no_source_text() -> None:
    """원문이 있는 페이지가 없으면 NO_SOURCE_TEXT로 건너뛰고 실패 페이지 수를 돌려준다."""
    repository = FakeRepository(make_target(failed_page_count=2))
    embedder = FakeEmbedder()

    assert run(repository, embedder) == ChunkIndexOutcome(
        status=ChunkIndexStatus.SKIPPED,
        reason=ChunkIndexSkipReason.NO_SOURCE_TEXT,
        failed_pages=2,
    )
    assert embedder.calls == []
    assert repository.replace_calls == []


def test_text_without_chunks_keeps_previous_index() -> None:
    """원문에서 조각이 나오지 않으면 임베딩도 교체도 하지 않고 NO_CHUNKS로 건너뛴다."""
    repository = FakeRepository(make_target("  \n "))
    embedder = FakeEmbedder()

    outcome = run(repository, embedder)

    assert outcome == ChunkIndexOutcome(
        status=ChunkIndexStatus.SKIPPED, reason=ChunkIndexSkipReason.NO_CHUNKS, succeeded_pages=1
    )
    assert embedder.calls == []
    assert repository.replace_calls == []


@pytest.mark.parametrize(
    "reason", [ChunkIndexSkipReason.STALE_ANALYSIS, ChunkIndexSkipReason.DELETED]
)
def test_replace_reason_is_returned(reason: ChunkIndexSkipReason) -> None:
    """교체 시점에 최신 분석이 아니거나 삭제됐으면 저장소가 준 이유로 건너뛴다."""
    repository = FakeRepository(make_target(SOURCE_TEXT), replace_result=reason)

    outcome = run(repository, FakeEmbedder())

    assert (outcome.status, outcome.reason, outcome.chunk_count) == (
        ChunkIndexStatus.SKIPPED,
        reason,
        0,
    )


def test_personal_info_lines_are_not_embedded() -> None:
    """개인정보 줄은 임베딩 입력과 저장 조각에서 빠지고 검사 결과 줄은 남는다."""
    page_text = "성명: 홍길동\n연락처 010-0000-0000\n혈색소 13.2 g/dL"
    repository = FakeRepository(make_target(page_text))
    embedder = FakeEmbedder()

    run(repository, embedder)

    assert embedder.calls == [["혈색소 13.2 g/dL"]]
    chunks, _, _ = repository.replace_calls[0]
    assert [chunk.content for chunk in chunks] == ["혈색소 13.2 g/dL"]


def test_only_personal_info_lines_skip_as_no_chunks() -> None:
    """개인정보 줄만 있으면 조각이 없어 기존 색인을 지우지 않고 NO_CHUNKS로 건너뛴다."""
    repository = FakeRepository(make_target("성명: 홍길동\n주소: 서울시 어딘가 1"))
    embedder = FakeEmbedder()

    outcome = run(repository, embedder)

    assert (outcome.status, outcome.reason) == (
        ChunkIndexStatus.SKIPPED,
        ChunkIndexSkipReason.NO_CHUNKS,
    )
    assert embedder.calls == []
    assert repository.replace_calls == []


def test_stored_chunks_fill_only_missing_embeddings() -> None:
    """조각이 이미 있으면 다시 나누지 않고 임베딩이 없는 조각만 임베딩해 채운다."""
    missing = (StoredChunk(uuid4(), "둘째 조각"), StoredChunk(uuid4(), "넷째 조각"))
    target = replace(make_target(SOURCE_TEXT), stored_chunk_count=4, missing_chunks=missing)
    repository = FakeRepository(target)
    embedder = FakeEmbedder()

    outcome = run(repository, embedder)

    assert embedder.calls == [["둘째 조각", "넷째 조각"]]
    assert repository.fill_calls == [([chunk.chunk_id for chunk in missing], 2, "fake-embedding")]
    assert repository.replace_calls == []
    assert (outcome.status, outcome.chunk_count, outcome.embedded_count) == (
        ChunkIndexStatus.INDEXED,
        4,
        2,
    )


def test_fully_indexed_chunks_need_no_embedding() -> None:
    """모든 조각에 임베딩이 있으면 임베더도 저장도 부르지 않고 INDEXED다."""
    target = replace(make_target(SOURCE_TEXT), stored_chunk_count=3)
    repository = FakeRepository(target)
    embedder = FakeEmbedder()

    outcome = run(repository, embedder)

    assert embedder.calls == []
    assert repository.fill_calls == []
    assert (outcome.status, outcome.chunk_count, outcome.embedded_count) == (
        ChunkIndexStatus.INDEXED,
        3,
        0,
    )


def test_stale_fill_is_skipped() -> None:
    """채우는 사이 새 분석이 생겼으면 저장소가 준 이유로 건너뛴다."""
    target = replace(
        make_target(SOURCE_TEXT),
        stored_chunk_count=1,
        missing_chunks=(StoredChunk(uuid4(), "조각"),),
    )
    repository = FakeRepository(target, replace_result=ChunkIndexSkipReason.STALE_ANALYSIS)

    outcome = run(repository, FakeEmbedder())

    assert (outcome.status, outcome.reason) == (
        ChunkIndexStatus.SKIPPED,
        ChunkIndexSkipReason.STALE_ANALYSIS,
    )


def test_embedding_failure_keeps_previous_chunks() -> None:
    """임베딩이 실패하면 교체를 부르지 않고 오류 코드와 함께 FAILED를 돌려준다."""
    repository = FakeRepository(make_target(SOURCE_TEXT))

    outcome = run(repository, FakeEmbedder(EmbeddingErrorCode.RATE_LIMITED))

    assert outcome == ChunkIndexOutcome(
        status=ChunkIndexStatus.FAILED,
        error_code=EmbeddingErrorCode.RATE_LIMITED.value,
        succeeded_pages=1,
    )
    assert repository.replace_calls == []
