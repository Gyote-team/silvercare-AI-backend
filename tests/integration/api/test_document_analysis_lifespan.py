"""서버 시작 때 남은 문서 분석 재실행을 설정값에 따라 실행기에 넣는지 검증한다."""

import threading
from concurrent.futures import Executor

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import app


class FakeClient:
    """LLM 클라이언트 대신 생성과 닫기만 받는 가짜."""

    def __init__(self, settings: Settings) -> None:
        pass

    def close(self) -> None:
        """닫을 것이 없다."""


@pytest.mark.parametrize(("enabled", "expected_calls"), [("true", 1), ("false", 0)])
def test_resume_is_submitted_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch, test_settings: Settings, enabled: str, expected_calls: int
) -> None:
    """켜져 있으면 재실행을 lifespan의 실행기로 한 번 넣고, 꺼져 있으면 넣지 않는다."""
    calls: list[Executor] = []
    called = threading.Event()

    def fake_resume(executor: Executor) -> None:
        calls.append(executor)
        called.set()

    monkeypatch.setenv("DOCUMENT_ANALYSIS_RESUME_ON_STARTUP", enabled)
    # test_settings는 conftest가 끈 값으로 이미 만들어졌다. 이 값만 빼고 환경 변수에서 다시 읽는다.
    values = test_settings.model_dump(exclude={"document_analysis_resume_on_startup"})
    monkeypatch.setattr("app.main.get_settings", lambda: Settings(**values))
    monkeypatch.setattr("app.main.OpenAiCompatibleLlmClient", FakeClient)
    monkeypatch.setattr("app.main.resume_unfinished_analyses", fake_resume)

    with TestClient(app):
        if expected_calls:
            assert called.wait(5)
        assert calls == [app.state.document_analysis_executor] * expected_calls
