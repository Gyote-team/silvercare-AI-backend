"""조각 색인 유스케이스 테스트. 저장소와 임베더는 가짜로 바꾼다."""

import logging
from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest

from app.application.ports.document_chunk_repository import ChunkIndexTarget
from app.application.ports.text_embedder import (
    EMBEDDING_DIMENSIONS,
    EmbeddingError,
    EmbeddingErrorCode,
)
from app.application.use_cases import index_document_chunks as use_case_module
from app.application.use_cases.index_document_chunks import (
    ChunkIndexOutcome,
    ChunkIndexStatus,
    index_document_chunks,
)
from app.domain.document.document_chunk import DocumentChunk, PageText

SOURCE_TEXT = "합성 검사 결과지\n혈색소 13.2 g/dL"


class FakeRepository:
    """대상 조회와 교체 호출만 흉내 내는 가짜 조각 저장소."""

    def __init__(self, target: ChunkIndexTarget | None, *, replaced: bool = True) -> None:
        self.target = target
        self.replaced = replaced
        self.replace_calls: list[tuple[list[DocumentChunk], list[list[float]], str]] = []

    def find_index_target(self, document_id: UUID) -> ChunkIndexTarget | None:
        """정해 둔 대상을 돌려준다."""
        return self.target

    def replace_chunks(
        self,
        target: ChunkIndexTarget,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
        model_name: str,
    ) -> bool:
        """받은 값을 기록하고 정해 둔 결과를 돌려준다."""
        self.replace_calls.append((list(chunks), [list(v) for v in vectors], model_name))
        return self.replaced

    def find_unindexed_document_ids(self, *, limit: int) -> list[UUID]:
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


def make_target(*texts: str) -> ChunkIndexTarget:
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
    )


def run(repository: FakeRepository, embedder: FakeEmbedder) -> ChunkIndexOutcome:
    """가짜로 유스케이스를 한 번 실행한다."""
    return index_document_chunks(uuid4(), repository=repository, embedder=embedder)


def test_chunks_are_embedded_and_replaced() -> None:
    """조각을 임베딩하고 모델 이름과 함께 교체를 한 번 부른다."""
    repository = FakeRepository(make_target(SOURCE_TEXT, "둘째 쪽"))
    embedder = FakeEmbedder()

    outcome = run(repository, embedder)

    assert outcome == ChunkIndexOutcome(status=ChunkIndexStatus.INDEXED, chunk_count=2)
    assert embedder.calls == [[SOURCE_TEXT, "둘째 쪽"]]
    chunks, vectors, model_name = repository.replace_calls[0]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert len(vectors) == 2
    assert model_name == "fake-embedding"


def test_missing_target_skips_without_embedding() -> None:
    """대상이 없으면 임베딩도 교체도 하지 않고 SKIPPED다."""
    repository = FakeRepository(None)
    embedder = FakeEmbedder()

    assert run(repository, embedder).status == ChunkIndexStatus.SKIPPED
    assert embedder.calls == []
    assert repository.replace_calls == []


def test_embedding_failure_keeps_previous_chunks(caplog: pytest.LogCaptureFixture) -> None:
    """임베딩이 실패하면 교체를 부르지 않고 코드만 로그에 남긴 채 FAILED를 돌려준다."""
    repository = FakeRepository(make_target(SOURCE_TEXT))

    with caplog.at_level(logging.ERROR, logger=use_case_module.__name__):
        outcome = run(repository, FakeEmbedder(EmbeddingErrorCode.RATE_LIMITED))

    assert outcome == ChunkIndexOutcome(
        status=ChunkIndexStatus.FAILED, error_code=EmbeddingErrorCode.RATE_LIMITED.value
    )
    assert repository.replace_calls == []
    assert "RATE_LIMITED" in caplog.text
    assert "혈색소" not in caplog.text


def test_superseded_replace_is_skipped() -> None:
    """교체 시점에 최신 분석이 아니면 SKIPPED다."""
    repository = FakeRepository(make_target(SOURCE_TEXT), replaced=False)

    assert run(repository, FakeEmbedder()).status == ChunkIndexStatus.SKIPPED


def test_text_without_content_replaces_with_nothing_and_skips_embedding() -> None:
    """원문이 공백뿐이면 임베딩을 부르지 않고 빈 조각으로 교체한다."""
    repository = FakeRepository(make_target("  \n "))
    embedder = FakeEmbedder()

    outcome = run(repository, embedder)

    assert outcome == ChunkIndexOutcome(status=ChunkIndexStatus.INDEXED, chunk_count=0)
    assert embedder.calls == []
    assert repository.replace_calls == [([], [], "fake-embedding")]
