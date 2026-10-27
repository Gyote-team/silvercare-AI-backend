"""서버가 허용한 원본 하나를 읽고 사용자 확인용 초안만 저장한다."""

import hashlib
from pathlib import Path

from app.application.ports.speech_transcriber import SpeechFailure, SpeechTranscriber
from app.application.ports.voice_job_queue import VoiceJob, VoiceJobQueue


class TranscribeVoiceUseCase:
    """개인·방문 범위는 Spring이 검증한 작업 행을 기준으로 고정한다."""

    def __init__(self, jobs: VoiceJobQueue, provider: SpeechTranscriber, root: str) -> None:
        self.jobs = jobs
        self.provider = provider
        self.root = Path(root).resolve()

    def run_once(self) -> bool:
        """작업이 있으면 하나 처리하고, 원문이나 공급자 오류 본문을 로그에 남기지 않는다."""
        job = self.jobs.claim()
        if job is None:
            return False
        try:
            path = self._path(job)
            draft = self.provider.transcribe(path)
            if not draft.strip() or len(draft) > 20000:
                raise SpeechFailure("TRANSCRIPT_INVALID")
            self.jobs.complete(job, draft, None)
        except SpeechFailure as failure:
            self.jobs.complete(job, None, failure.code)
        except (OSError, ValueError):
            self.jobs.complete(job, None, "AUDIO_UNAVAILABLE")
        except Exception:
            self.jobs.complete(job, None, "STT_UNAVAILABLE")
        return True

    def _path(self, job: VoiceJob) -> Path:
        expected = Path("voice") / str(job.patient_id)
        relative = Path(job.storage_key)
        if relative.parent != expected or relative.stem != str(job.id):
            raise SpeechFailure("AUDIO_INVALID")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise SpeechFailure("AUDIO_UNAVAILABLE")
        if path.stat().st_size != job.size_bytes or not 0 < job.size_bytes <= 20 * 1024 * 1024:
            raise SpeechFailure("AUDIO_INVALID")
        if hashlib.sha256(path.read_bytes()).hexdigest() != job.content_hash:
            raise SpeechFailure("AUDIO_INVALID")
        return path
