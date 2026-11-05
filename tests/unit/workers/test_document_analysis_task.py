"""문서 분석 백그라운드 작업의 실패 기록과 로그를 검증한다. DB와 VLM은 가짜로 바꾼다."""

import logging
from datetime import datetime, timedelta
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr

from app.application.ports.document_analysis_repository import AnalysisTarget
from app.application.ports.document_chunk_repository import (
    ChunkIndexSkipReason,
    ChunkIndexTarget,
)
from app.application.ports.text_embedder import EmbeddingError, EmbeddingErrorCode
from app.application.use_cases.analyze_document import DocumentAnalysisResult
from app.application.use_cases.index_document_chunks import (
    ChunkIndexOutcome,
    ChunkIndexStatus,
    index_document_chunks,
)
from app.core.config import Settings
from app.domain.document.document_analysis import AnalysisFailure, DocumentStatus
from app.domain.document.document_chunk import PageText
from app.workers import document_analysis_task

_REQUEST_ID = "req-test-1"
_OBJECT_KEY = "documents/synthetic-test.pdf"
_MIME_TYPE = "application/pdf"


class FakeRepository:
    """대상 조회와 시작 전 실패 기록만 흉내 내는 가짜 저장소."""

    def __init__(self, session_factory: object, *, stale_after: timedelta) -> None:
        self.target: AnalysisTarget | None = None
        self.before_start_failures: list[AnalysisFailure] = []
        self.before_start_error: Exception | None = None
        self.resumable_ids: list[UUID] = []
        self.resumable_error: Exception | None = None
        self.unindexed_ids: list[UUID] = []
        self.indexed_ids: list[UUID] = []
        self.index_error: Exception | None = None
        self.index_outcome = ChunkIndexOutcome(status=ChunkIndexStatus.INDEXED)

    def find_unindexed_document_ids(self, *, model_name: str, limit: int) -> list[UUID]:
        """조각 저장소 대신 정해 둔 미색인 문서 id를 돌려준다. 받은 모델 이름을 기록한다."""
        self.unindexed_model_name = model_name
        return self.unindexed_ids

    def index(self, document_id: UUID, **kwargs: object) -> ChunkIndexOutcome:
        """index_document_chunks 대신 호출을 기록하고 정해 둔 결과를 돌려준다."""
        if self.index_error is not None:
            raise self.index_error
        self.indexed_ids.append(document_id)
        return self.index_outcome

    def find_target(self, document_id: UUID) -> AnalysisTarget | None:
        """정해 둔 대상을 돌려준다."""
        return self.target

    def find_resumable_document_ids(
        self, now: datetime, *, pending_after: timedelta, limit: int
    ) -> list[UUID]:
        """정해 둔 id를 돌려준다. 정해 둔 예외가 있으면 던진다."""
        if self.resumable_error is not None:
            raise self.resumable_error
        return self.resumable_ids

    def mark_failed_before_start(
        self, document_id: UUID, failure: AnalysisFailure, completed_at: datetime
    ) -> bool:
        """받은 실패 기록을 남긴다. 정해 둔 예외가 있으면 던진다."""
        if self.before_start_error is not None:
            raise self.before_start_error
        self.before_start_failures.append(failure)
        return True


class FakeExtractor:
    """닫기만 흉내 내는 가짜 추출기. 정해 둔 예외가 있으면 닫을 때 던진다."""

    def __init__(self, close_error: Exception | None = None) -> None:
        self.close_error = close_error

    def close(self) -> None:
        """정해 둔 예외가 있으면 던진다."""
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture
def repository(monkeypatch: pytest.MonkeyPatch, test_settings: Settings) -> FakeRepository:
    """작업이 만드는 설정·세션·저장소·추출기·유스케이스를 가짜로 바꾸고 가짜 저장소를 돌려준다."""
    fake = FakeRepository(None, stale_after=timedelta())
    # .env 값과 상관없이 임베딩 키가 있는 설정으로 고정한다. 키 없는 경우는 테스트에서 바꾼다.
    settings = test_settings.model_copy(update={"embedding_api_key": SecretStr("sk-test")})
    monkeypatch.setattr(document_analysis_task, "get_settings", lambda: settings)
    monkeypatch.setattr(document_analysis_task, "get_session_factory", lambda: None)
    monkeypatch.setattr(
        document_analysis_task, "SqlAlchemyDocumentAnalysisRepository", lambda *a, **k: fake
    )
    monkeypatch.setattr(document_analysis_task, "create_page_extractor", lambda _: FakeExtractor())
    monkeypatch.setattr(
        document_analysis_task,
        "analyze_document",
        lambda *a, **k: DocumentAnalysisResult(document_status=None),
    )
    # 조각 저장소와 임베더, 색인 유스케이스도 가짜로 바꾼다. 실제 키와 DB를 쓰지 않는다.
    monkeypatch.setattr(
        document_analysis_task, "SqlAlchemyDocumentChunkRepository", lambda *a, **k: fake
    )
    monkeypatch.setattr(document_analysis_task, "create_text_embedder", lambda _: FakeExtractor())
    monkeypatch.setattr(document_analysis_task, "index_document_chunks", fake.index)
    return fake


def _run(document_id: UUID) -> None:
    """요청 값으로 작업을 한 번 실행한다."""
    document_analysis_task.run_document_analysis(document_id, _REQUEST_ID, _OBJECT_KEY, _MIME_TYPE)


def _raise_with_object_key(*args: object, **kwargs: object) -> None:
    """메시지에 objectKey를 담은 예외를 던진다. 로그로 새지 않는지 확인하는 데 쓴다."""
    raise RuntimeError(_OBJECT_KEY)


def test_setup_failure_is_recorded_before_start(
    monkeypatch: pytest.MonkeyPatch, repository: FakeRepository
) -> None:
    """추출기를 만들다 실패하면 시작 전 실패를 retryable로 한 번 기록한다."""
    monkeypatch.setattr(document_analysis_task, "create_page_extractor", _raise_with_object_key)

    _run(uuid4())

    assert repository.before_start_failures == [document_analysis_task._SETUP_FAILURE]
    assert repository.before_start_failures[0].retryable is True


def test_failure_log_has_ids_and_location_but_not_message(
    monkeypatch: pytest.MonkeyPatch,
    repository: FakeRepository,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """실패 로그에는 식별자, 예외 이름, 파일 이름과 줄 번호만 남고 예외 메시지는 남지 않는다."""
    monkeypatch.setattr(document_analysis_task, "analyze_document", _raise_with_object_key)
    document_id = uuid4()

    with caplog.at_level(logging.ERROR, logger=document_analysis_task.__name__):
        _run(document_id)

    assert str(document_id) in caplog.text
    assert _REQUEST_ID in caplog.text
    assert "RuntimeError" in caplog.text
    assert "test_document_analysis_task.py:" in caplog.text
    assert _OBJECT_KEY not in caplog.text


def test_settings_failure_is_logged_without_recording(
    monkeypatch: pytest.MonkeyPatch,
    repository: FakeRepository,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """설정을 읽지 못해 저장소를 만들 수 없으면 기록하지 않고 로그만 남긴다."""
    monkeypatch.setattr(document_analysis_task, "get_settings", _raise_with_object_key)

    with caplog.at_level(logging.ERROR, logger=document_analysis_task.__name__):
        _run(uuid4())

    assert repository.before_start_failures == []
    assert "문서 분석 작업 실패" in caplog.text


def test_failure_while_recording_is_logged_by_class_name(
    monkeypatch: pytest.MonkeyPatch,
    repository: FakeRepository,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """시작 전 실패 기록까지 실패하면 예외를 밖으로 던지지 않고 클래스 이름만 남긴다."""
    monkeypatch.setattr(document_analysis_task, "create_page_extractor", _raise_with_object_key)
    repository.before_start_error = ConnectionError(_OBJECT_KEY)

    with caplog.at_level(logging.ERROR, logger=document_analysis_task.__name__):
        _run(uuid4())

    assert "시작 전 실패 기록 중 오류" in caplog.text
    assert "ConnectionError" in caplog.text
    assert _OBJECT_KEY not in caplog.text


def test_close_failure_is_logged_by_class_name(
    monkeypatch: pytest.MonkeyPatch,
    repository: FakeRepository,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """추출기를 닫다 실패해도 예외를 밖으로 던지지 않고 클래스 이름만 남긴다."""
    monkeypatch.setattr(
        document_analysis_task,
        "create_page_extractor",
        lambda _: FakeExtractor(close_error=OSError(_OBJECT_KEY)),
    )

    with caplog.at_level(logging.ERROR, logger=document_analysis_task.__name__):
        _run(uuid4())

    assert "추출기 닫기 실패" in caplog.text
    assert "OSError" in caplog.text
    assert _OBJECT_KEY not in caplog.text


class FakeExecutor:
    """제출한 함수와 인자만 기록하는 가짜 실행기."""

    def __init__(self) -> None:
        self.submitted: list[tuple[object, ...]] = []

    def submit(self, fn: object, *args: object) -> None:
        """실행하지 않고 기록만 한다."""
        self.submitted.append((fn, *args))


def test_resume_submits_each_document_once(repository: FakeRepository) -> None:
    """조회한 문서마다 요청 값 없이 분석을 한 번씩 제출한다."""
    first, second = uuid4(), uuid4()
    repository.resumable_ids = [first, second]
    executor = FakeExecutor()

    document_analysis_task.resume_unfinished_analyses(executor)

    run = document_analysis_task.run_document_analysis
    assert executor.submitted == [(run, first, None), (run, second, None)]


def test_resume_lookup_failure_is_logged_without_submit(
    repository: FakeRepository, caplog: pytest.LogCaptureFixture
) -> None:
    """조회가 실패하면 제출하지 않고 클래스 이름만 로그로 남긴다."""
    repository.resumable_error = ConnectionError(_OBJECT_KEY)
    executor = FakeExecutor()

    with caplog.at_level(logging.ERROR, logger=document_analysis_task.__name__):
        document_analysis_task.resume_unfinished_analyses(executor)

    assert executor.submitted == []
    assert "ConnectionError" in caplog.text
    assert _OBJECT_KEY not in caplog.text


def test_resumed_run_does_not_compare_request_values(
    repository: FakeRepository, caplog: pytest.LogCaptureFixture
) -> None:
    """재실행은 요청 값이 없으므로 대상이 있어도 비교 경고를 남기지 않는다."""
    document_id = uuid4()
    repository.target = AnalysisTarget(
        document_id=document_id,
        analysis_id=uuid4(),
        storage_key="documents/synthetic-stored.pdf",
        mime_type=_MIME_TYPE,
    )

    with caplog.at_level(logging.WARNING, logger=document_analysis_task.__name__):
        document_analysis_task.run_document_analysis(document_id, None)

    assert "요청 값이 DB와 다릅니다" not in caplog.text


def test_request_differing_from_db_logs_field_names_only(
    repository: FakeRepository, caplog: pytest.LogCaptureFixture
) -> None:
    """요청의 objectKey가 DB와 다르면 필드 이름만 warning으로 남기고 값은 남기지 않는다."""
    document_id = uuid4()
    stored_key = "documents/synthetic-stored.pdf"
    repository.target = AnalysisTarget(
        document_id=document_id,
        analysis_id=uuid4(),
        storage_key=stored_key,
        mime_type=_MIME_TYPE,
    )

    with caplog.at_level(logging.WARNING, logger=document_analysis_task.__name__):
        _run(document_id)

    assert "fields=objectKey" in caplog.text
    assert "mimeType" not in caplog.text
    assert _OBJECT_KEY not in caplog.text
    assert stored_key not in caplog.text


def _finish_analysis_with(monkeypatch: pytest.MonkeyPatch, status: DocumentStatus | None) -> None:
    """분석이 주어진 문서 상태로 끝난 것처럼 바꾼다."""
    monkeypatch.setattr(
        document_analysis_task,
        "analyze_document",
        lambda *a, **k: DocumentAnalysisResult(document_status=status),
    )


@pytest.mark.parametrize(
    ("status", "indexed"),
    [
        (DocumentStatus.READY, True),
        (DocumentStatus.NEEDS_REVIEW, True),
        (DocumentStatus.FAILED, False),
        (None, False),
    ],
)
def test_indexing_runs_only_after_successful_analysis(
    monkeypatch: pytest.MonkeyPatch,
    repository: FakeRepository,
    status: DocumentStatus | None,
    indexed: bool,
) -> None:
    """분석이 READY나 NEEDS_REVIEW로 끝났을 때만 같은 실행에서 색인한다."""
    _finish_analysis_with(monkeypatch, status)
    document_id = uuid4()

    _run(document_id)

    assert repository.indexed_ids == ([document_id] if indexed else [])


def test_indexing_failure_is_logged_without_recording(
    monkeypatch: pytest.MonkeyPatch,
    repository: FakeRepository,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """색인이 실패해도 실패 기록을 남기지 않고, 로그에는 id와 예외 이름만 남는다."""
    _finish_analysis_with(monkeypatch, DocumentStatus.READY)
    repository.index_error = RuntimeError(_OBJECT_KEY)
    document_id = uuid4()

    with caplog.at_level(logging.ERROR, logger=document_analysis_task.__name__):
        _run(document_id)

    assert repository.before_start_failures == []
    assert "조각 색인 작업 실패" in caplog.text
    assert str(document_id) in caplog.text
    assert "RuntimeError" in caplog.text
    assert _OBJECT_KEY not in caplog.text


def test_index_outcome_is_logged_with_reason_and_counts(
    repository: FakeRepository, caplog: pytest.LogCaptureFixture
) -> None:
    """색인 결과는 상태, 이유, 조각 수, 새로 임베딩한 수, 성공·실패 페이지 수와 함께 남는다."""
    repository.index_outcome = ChunkIndexOutcome(
        status=ChunkIndexStatus.INDEXED,
        chunk_count=5,
        embedded_count=2,
        succeeded_pages=2,
        failed_pages=1,
    )
    document_id = uuid4()

    with caplog.at_level(logging.INFO, logger=document_analysis_task.__name__):
        document_analysis_task.run_chunk_indexing(document_id, _REQUEST_ID)

    [record] = [r for r in caplog.records if "조각 색인 결과" in r.getMessage()]
    assert record.levelno == logging.INFO
    message = record.getMessage()
    assert str(document_id) in message
    assert "status=INDEXED" in message
    assert "reason=None" in message
    assert "chunks=5, embedded=2, succeeded_pages=2, failed_pages=1" in message


def test_skipped_index_is_logged_with_reason(
    repository: FakeRepository, caplog: pytest.LogCaptureFixture
) -> None:
    """건너뛴 색인은 이유와 함께 INFO로 남는다."""
    repository.index_outcome = ChunkIndexOutcome(
        status=ChunkIndexStatus.SKIPPED, reason=ChunkIndexSkipReason.NO_CHUNKS
    )

    with caplog.at_level(logging.INFO, logger=document_analysis_task.__name__):
        document_analysis_task.run_chunk_indexing(uuid4(), _REQUEST_ID)

    [record] = [r for r in caplog.records if "조각 색인 결과" in r.getMessage()]
    assert record.levelno == logging.INFO
    assert "status=SKIPPED, reason=NO_CHUNKS" in record.getMessage()


class _SourceTextRepository:
    """원문이 담긴 색인 대상을 돌려주는 가짜 조각 저장소. 저장은 하지 않는다."""

    source_text = "합성 검사 결과지\n혈색소 13.2 g/dL"

    def find_index_target(self, document_id: UUID, model_name: str) -> ChunkIndexTarget:
        """원문 한 쪽과 실패 한 쪽이 있는 대상을 돌려준다."""
        return ChunkIndexTarget(
            document_id=document_id,
            analysis_id=uuid4(),
            patient_id=uuid4(),
            visit_id=uuid4(),
            pages=(PageText(page_id=uuid4(), page_no=1, text=self.source_text),),
            failed_page_count=1,
        )


class _RateLimitedEmbedder:
    """호출 한도 초과로 실패하는 가짜 임베더. 외부 API를 부르지 않는다."""

    model_name = "fake-embedding"

    def embed(self, texts: object) -> list[list[float]]:
        """항상 RATE_LIMITED로 실패한다."""
        raise EmbeddingError(EmbeddingErrorCode.RATE_LIMITED)

    def close(self) -> None:
        """닫을 것이 없다."""


def test_failed_index_logs_error_code_without_source_text(
    monkeypatch: pytest.MonkeyPatch,
    repository: FakeRepository,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """임베딩이 실패하면 오류 코드는 ERROR로 남고 문서 원문은 로그에 남지 않는다."""
    chunk_repository = _SourceTextRepository()
    monkeypatch.setattr(document_analysis_task, "index_document_chunks", index_document_chunks)
    monkeypatch.setattr(
        document_analysis_task, "SqlAlchemyDocumentChunkRepository", lambda *a: chunk_repository
    )
    monkeypatch.setattr(
        document_analysis_task, "create_text_embedder", lambda _: _RateLimitedEmbedder()
    )
    document_id = uuid4()

    with caplog.at_level(logging.DEBUG):
        document_analysis_task.run_chunk_indexing(document_id, _REQUEST_ID)

    [record] = [r for r in caplog.records if "조각 색인 결과" in r.getMessage()]
    assert record.levelno == logging.ERROR
    assert "status=FAILED" in record.getMessage()
    assert "error_code=RATE_LIMITED" in record.getMessage()
    assert "embedded=0, succeeded_pages=1, failed_pages=1" in record.getMessage()
    assert "혈색소" not in caplog.text
    assert "합성 검사 결과지" not in caplog.text


def test_resume_submits_unindexed_documents_after_analyses(repository: FakeRepository) -> None:
    """시작 시 재실행은 남은 분석 다음에 임베딩이 없는 문서의 색인을 넣는다."""
    analysis_id, unindexed_id = uuid4(), uuid4()
    repository.resumable_ids = [analysis_id]
    repository.unindexed_ids = [unindexed_id]
    executor = FakeExecutor()

    document_analysis_task.resume_unfinished_analyses(executor)

    assert executor.submitted == [
        (document_analysis_task.run_document_analysis, analysis_id, None),
        (document_analysis_task.run_chunk_indexing, unindexed_id, None),
    ]
    # 색인됨 판단은 설정의 임베딩 모델 기준이다.
    assert repository.unindexed_model_name == document_analysis_task.get_settings().embedding_model


def test_resume_indexes_even_when_analysis_lookup_fails(repository: FakeRepository) -> None:
    """남은 분석 조회가 실패해도 미색인 문서의 색인은 넣는다."""
    repository.resumable_error = ConnectionError(_OBJECT_KEY)
    unindexed_id = uuid4()
    repository.unindexed_ids = [unindexed_id]
    executor = FakeExecutor()

    document_analysis_task.resume_unfinished_analyses(executor)

    assert executor.submitted == [(document_analysis_task.run_chunk_indexing, unindexed_id, None)]


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_resume_skips_indexing_without_embedding_key(
    monkeypatch: pytest.MonkeyPatch,
    repository: FakeRepository,
    test_settings: Settings,
    caplog: pytest.LogCaptureFixture,
    api_key: str | None,
) -> None:
    """EMBEDDING_API_KEY가 없거나 비어 있으면 미색인 문서를 넣지 않고 info 로그 한 줄만 남긴다."""
    settings = test_settings.model_copy(
        update={"embedding_api_key": SecretStr(api_key) if api_key is not None else None}
    )
    monkeypatch.setattr(document_analysis_task, "get_settings", lambda: settings)
    analysis_id = uuid4()
    repository.resumable_ids = [analysis_id]
    repository.unindexed_ids = [uuid4()]
    executor = FakeExecutor()

    with caplog.at_level(logging.INFO, logger=document_analysis_task.__name__):
        document_analysis_task.resume_unfinished_analyses(executor)

    assert executor.submitted == [(document_analysis_task.run_document_analysis, analysis_id, None)]
    skipped = [record for record in caplog.records if "EMBEDDING_API_KEY" in record.getMessage()]
    assert len(skipped) == 1
    assert skipped[0].levelno == logging.INFO
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
