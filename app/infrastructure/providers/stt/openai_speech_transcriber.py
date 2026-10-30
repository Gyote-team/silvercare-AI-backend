"""OpenAI 호환 파일 전사 API를 호출한다. 초안을 자동 확정하지 않는다."""

from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from app.application.ports.speech_transcriber import SpeechFailure
from app.core.config import Settings


class OpenAiSpeechTranscriber:
    """공급자 키와 모델을 STT 전용 설정으로 분리한다."""

    def __init__(self, settings: Settings, client=None) -> None:
        key = next((value for value in (settings.stt_api_key, settings.openai_api_key)
                    if value and value.get_secret_value().strip()), None)
        if client is None and key is None:
            raise SpeechFailure("STT_NOT_CONFIGURED")
        self.model = settings.stt_model
        self.client = client or OpenAI(
            api_key=key.get_secret_value(),
            base_url=settings.stt_base_url,
            timeout=settings.stt_timeout_seconds,
            max_retries=0,
        )

    def transcribe(self, path: Path) -> str:
        """음성을 한국어 텍스트로 변환하고 빈 인식·과도한 길이는 실패로 처리한다."""
        language_options = (
            {"extra_body": {"languages": ["ko"]}}
            if self.model == "gpt-transcribe" else {"language": "ko"}
        )
        try:
            with path.open("rb") as audio:
                result = self.client.audio.transcriptions.create(
                    model=self.model, file=audio, response_format="json", **language_options
                )
        except APITimeoutError:
            raise SpeechFailure("STT_TIMEOUT") from None
        except APIConnectionError:
            raise SpeechFailure("STT_UNAVAILABLE") from None
        except APIStatusError as error:
            code = {
                400: "AUDIO_INVALID", 401: "STT_AUTH_FAILED", 403: "STT_AUTH_FAILED",
                413: "AUDIO_TOO_LARGE", 429: "STT_RATE_LIMIT",
            }.get(error.status_code, "STT_UNAVAILABLE")
            raise SpeechFailure(code) from None
        text = result.text.strip()
        if not text:
            raise SpeechFailure("NO_SPEECH")
        if len(text) > 20000:
            raise SpeechFailure("TRANSCRIPT_TOO_LONG")
        return text

    def close(self) -> None:
        """작업자 종료 시 연결 풀을 닫는다."""
        self.client.close()
