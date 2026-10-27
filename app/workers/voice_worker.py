"""python -m app.workers.voice_worker로 영속 음성 큐를 처리한다."""

import time

from app.application.use_cases.transcribe_voice import TranscribeVoiceUseCase
from app.core.config import get_settings
from app.infrastructure.db.repositories.voice_job_repository import VoiceJobRepository
from app.infrastructure.db.session import get_engine
from app.infrastructure.providers.stt.openai_speech_transcriber import OpenAiSpeechTranscriber


def main() -> None:
    """키가 없는 환경은 시작 시 명확히 실패시키고 원본을 외부 전송하지 않는다."""
    settings = get_settings()
    provider = OpenAiSpeechTranscriber(settings)
    use_case = TranscribeVoiceUseCase(
        VoiceJobRepository(get_engine()), provider, settings.storage_local_root
    )
    try:
        while True:
            if not use_case.run_once():
                time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        provider.close()


if __name__ == "__main__":
    main()
