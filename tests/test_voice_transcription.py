"""공급자 호출과 허용된 파일 범위·실패·늦은 결과 처리의 회귀 검사."""

import hashlib
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.application.ports.speech_transcriber import SpeechFailure
from app.application.ports.voice_job_queue import VoiceJob
from app.application.use_cases.transcribe_voice import TranscribeVoiceUseCase
from app.core.config import Settings
from app.infrastructure.providers.stt.openai_speech_transcriber import OpenAiSpeechTranscriber


@pytest.fixture
def voice(tmp_path):
    """의료정보가 없는 로컬 테스트 파일을 준비한다."""
    job_id, patient_id = uuid4(), uuid4()
    data = b"RIFF" + b"\x00" * 40
    key = f"voice/{patient_id}/{job_id}.wav"
    path = tmp_path / key
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    return VoiceJob(job_id, patient_id, key, hashlib.sha256(data).hexdigest(), len(data), uuid4())


def test_worker_only_saves_draft(voice, tmp_path):
    jobs, provider = Mock(), Mock()
    jobs.claim.return_value = voice
    provider.transcribe.return_value = "확인 전 초안"
    worker = TranscribeVoiceUseCase(jobs, provider, str(tmp_path))
    assert worker.run_once()
    jobs.complete.assert_called_once_with(voice, "확인 전 초안", None)


@pytest.mark.parametrize("key", ["../../secret.wav", "/secret.wav", "voice/other/file.wav"])
def test_worker_rejects_out_of_scope_key(voice, tmp_path, key):
    jobs, provider = Mock(), Mock()
    job = replace(voice, storage_key=key)
    jobs.claim.return_value = job
    TranscribeVoiceUseCase(jobs, provider, str(tmp_path)).run_once()
    provider.transcribe.assert_not_called()
    jobs.complete.assert_called_once_with(job, None, "AUDIO_INVALID")


def test_worker_rejects_changed_original(voice, tmp_path):
    jobs, provider = Mock(), Mock()
    jobs.claim.return_value = replace(voice, content_hash="0" * 64)
    TranscribeVoiceUseCase(jobs, provider, str(tmp_path)).run_once()
    provider.transcribe.assert_not_called()
    assert jobs.complete.call_args.args[2] == "AUDIO_INVALID"


def test_failure_has_safe_code_not_provider_message(voice, tmp_path):
    jobs, provider = Mock(), Mock()
    jobs.claim.return_value = voice
    provider.transcribe.side_effect = RuntimeError("private provider content")
    TranscribeVoiceUseCase(jobs, provider, str(tmp_path)).run_once()
    jobs.complete.assert_called_once_with(voice, None, "STT_UNAVAILABLE")


def test_no_job_does_not_call_provider(tmp_path):
    jobs, provider = Mock(), Mock()
    jobs.claim.return_value = None
    assert not TranscribeVoiceUseCase(jobs, provider, str(tmp_path)).run_once()
    provider.transcribe.assert_not_called()


def settings():
    """테스트에서 외부 네트워크나 실제 키를 사용하지 않는다."""
    return Settings(database_url="sqlite://", ai_internal_token="test", storage_local_root=".")


def test_provider_requires_key():
    with pytest.raises(SpeechFailure, match="STT_NOT_CONFIGURED"):
        OpenAiSpeechTranscriber(settings())


@pytest.mark.parametrize(
    "content,code", [("   ", "NO_SPEECH"), ("가" * 20001, "TRANSCRIPT_TOO_LONG")],
    ids=["empty-speech", "too-long"],
)
def test_provider_does_not_invent_or_truncate_text(tmp_path, content, code):
    path = tmp_path / "test.wav"
    path.write_bytes(b"test")
    client = Mock()
    client.audio.transcriptions.create.return_value = SimpleNamespace(text=content)
    with pytest.raises(SpeechFailure, match=code):
        OpenAiSpeechTranscriber(settings(), client).transcribe(path)


def test_provider_sends_file_and_korean_language(tmp_path):
    path = tmp_path / "test.wav"
    path.write_bytes(b"test")
    client = Mock()
    client.audio.transcriptions.create.return_value = SimpleNamespace(text=" 인식 내용 ")
    assert OpenAiSpeechTranscriber(settings(), client).transcribe(path) == "인식 내용"
    assert client.audio.transcriptions.create.call_args.kwargs["language"] == "ko"
