"""문서 분석 백그라운드 작업의 실패 기록과 로그를 검증한다. DB와 VLM은 가짜로 바꾼다."""

import logging
from datetime import datetime, timedelta
from uuid import UUID, uuid4

import pytest

from app.application.ports.document_analysis_repository import AnalysisTarget
from app.application.use_cases.analyze_document import DocumentAnalysisResult
from app.core.config import Settings
from app.domain.document.document_analysis import AnalysisFailure
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

    def find_target(self, document_id: UUID) -> AnalysisTarget | None:
        """정해 둔 대상을 돌려준다."""
        return self.target

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
    monkeypatch.setattr(document_analysis_task, "get_settings", lambda: test_settings)
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
