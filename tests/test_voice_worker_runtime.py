"""작업자 DB 장애 복구와 프로세스 전체 시간 제한을 외부 API 없이 검증한다."""

import logging
import multiprocessing
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

import pytest
from pydantic import SecretStr, ValidationError
from sqlalchemy.exc import OperationalError

from app.application.ports.speech_transcriber import SpeechFailure
from app.core.config import Settings
from app.infrastructure.providers.stt.openai_speech_transcriber import OpenAiSpeechTranscriber
from app.workers.voice_worker import run_loop


def config(**overrides):
    """로컬 실제 키·.env가 테스트에 섞이지 않도록 명시한다."""
    values = dict(database_url="sqlite://", ai_internal_token="test", storage_local_root=".",
                  stt_api_key=None, openai_api_key=None, stt_model="gpt-transcribe",
                  stt_timeout_seconds=180, stt_lease_seconds=600)
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_document_key_is_not_used_for_stt():
    with pytest.raises(SpeechFailure, match="STT_NOT_CONFIGURED"):
        OpenAiSpeechTranscriber(config(openai_api_key=SecretStr("document-only-key")))


@pytest.mark.parametrize("lease", [180, 210])
def test_lease_must_exceed_total_timeout_and_shutdown_margin(lease):
    with pytest.raises(ValidationError, match="STT_LEASE_SECONDS"):
        config(stt_lease_seconds=lease)


def test_db_failure_logs_safe_code_and_loop_recovers(monkeypatch, caplog):
    use_case = Mock()
    use_case.run_once.side_effect = [
        OperationalError("private SQL", {}, Exception("private connection detail")),
        True, KeyboardInterrupt,
    ]
    sleeps = []
    monkeypatch.setattr("app.workers.voice_worker.time.sleep", sleeps.append)
    with caplog.at_level(logging.INFO), pytest.raises(KeyboardInterrupt):
        run_loop(use_case, 5)
    assert use_case.run_once.call_count == 3
    assert sleeps == [5]
    assert "VOICE_DB_UNAVAILABLE" in caplog.text
    assert "private" not in caplog.text


def test_long_transcript_is_preserved_and_limit_is_configurable(tmp_path):
    path = tmp_path / "test.wav"
    path.write_bytes(b"test")
    client = Mock()
    long_text = "가" * 20001
    client.audio.transcriptions.create.return_value.text = long_text
    assert OpenAiSpeechTranscriber(config(), client).transcribe(path) == long_text
    with pytest.raises(SpeechFailure, match="TRANSCRIPT_TOO_LONG"):
        OpenAiSpeechTranscriber(config(stt_max_transcript_chars=20000), client).transcribe(path)


@pytest.mark.parametrize("slow", [False, True], ids=["success", "total-deadline"])
def test_actual_child_process_returns_or_is_stopped_at_total_deadline(tmp_path, slow):
    """읽기 제한보다 짧게 데이터를 계속 보내도 전체 기한을 넘겨 실행되지 않는다."""
    received = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            received.set()
            payload = b'{"text":"local transcript"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                for byte in payload:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    if slow:
                        time.sleep(0.25)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    path = tmp_path / "test.wav"
    path.write_bytes(b"test-audio")
    provider = OpenAiSpeechTranscriber(config(
        stt_api_key=SecretStr("local-test-key"),
        stt_base_url=f"http://127.0.0.1:{server.server_port}/v1",
        stt_timeout_seconds=3 if slow else 10,
    ))
    before = {p.pid for p in multiprocessing.active_children()}
    started = time.monotonic()
    try:
        if slow:
            with pytest.raises(SpeechFailure, match="STT_TIMEOUT"):
                provider.transcribe(path)
            assert time.monotonic() - started < 6
        else:
            assert provider.transcribe(path) == "local transcript"
        assert received.is_set()
        assert {p.pid for p in multiprocessing.active_children()} == before
    finally:
        provider.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
