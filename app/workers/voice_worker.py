"""python -m app.workers.voice_worker로 영속 음성 큐를 처리한다."""

import logging
import time

from sqlalchemy.exc import SQLAlchemyError

from app.application.use_cases.transcribe_voice import TranscribeVoiceUseCase
from app.core.config import get_settings
from app.infrastructure.db.repositories.voice_job_repository import VoiceJobRepository
from app.infrastructure.db.session import get_engine
from app.infrastructure.providers.stt.openai_speech_transcriber import OpenAiSpeechTranscriber

logger = logging.getLogger(__name__)


def run_loop(use_case, retry_seconds: float) -> None:
    """DB 장애는 원문·접속 문자열 없이 기록하고 기다린 뒤 다시 조회한다."""
    while True:
        try:
            if not use_case.run_once():
                time.sleep(2)
        except SQLAlchemyError as error:
            logger.warning("VOICE_DB_UNAVAILABLE error_type=%s", type(error).__name__)
            time.sleep(retry_seconds)


def main() -> None:
    """키가 없는 환경은 시작 시 명확히 실패시키고 원본을 외부 전송하지 않는다."""
    settings = get_settings()
    provider = OpenAiSpeechTranscriber(settings)
    use_case = TranscribeVoiceUseCase(
        VoiceJobRepository(get_engine(), settings.stt_lease_seconds), provider,
        settings.storage_local_root, settings.stt_max_transcript_chars,
    )
    logging.basicConfig(level=logging.INFO)
    logger.info("VOICE_WORKER_STARTED lease_seconds=%s", settings.stt_lease_seconds)
    try:
        run_loop(use_case, settings.stt_db_retry_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        provider.close()
        logger.info("VOICE_WORKER_STOPPED")


if __name__ == "__main__":
    main()
