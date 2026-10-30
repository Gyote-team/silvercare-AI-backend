"""OpenAI 호환 파일 전사 API를 호출한다. 초안을 자동 확정하지 않는다."""

import multiprocessing
import time
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from app.application.ports.speech_transcriber import SpeechFailure
from app.core.config import Settings


class OpenAiSpeechTranscriber:
    """공급자 키와 모델을 STT 전용 설정으로 분리한다."""

    def __init__(self, settings: Settings, client=None) -> None:
        key = settings.stt_api_key
        if client is None and (key is None or not key.get_secret_value().strip()):
            raise SpeechFailure("STT_NOT_CONFIGURED")
        self.model = settings.stt_model
        self.client = client
        self.key = key
        self.base_url = settings.stt_base_url
        self.timeout = settings.stt_timeout_seconds
        self.max_chars = settings.stt_max_transcript_chars

    def transcribe(self, path: Path) -> str:
        """음성을 한국어 텍스트로 변환하고 빈 인식·과도한 길이는 실패로 처리한다."""
        if self.client is not None:
            return _request(self.client, path, self.model, self.max_chars)
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_transcribe_child, args=(
            sender, str(path), self.key.get_secret_value(), self.base_url,
            self.model, self.timeout, self.max_chars,
        ))
        started = time.monotonic()
        try:
            process.start()
            sender.close()
            remaining = max(0, self.timeout - (time.monotonic() - started))
            if not receiver.poll(remaining):
                raise SpeechFailure("STT_TIMEOUT")
            code, result = receiver.recv()
            if code:
                raise SpeechFailure(code)
            return result
        except (EOFError, OSError):
            raise SpeechFailure("STT_UNAVAILABLE") from None
        finally:
            receiver.close()
            sender.close()
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
                if process.is_alive():
                    # 종료에 실패한 채 다음 작업을 가져가지는 않는다.
                    raise SystemExit("STT child process could not be stopped")
                process.close()

    def close(self) -> None:
        """테스트 주입 클라이언트의 연결을 닫는다. 운영 클라이언트는 자식이 소유한다."""
        if self.client is not None:
            self.client.close()


def _transcribe_child(sender, path, key, base_url, model, timeout, max_chars) -> None:
    """취소 가능한 별도 프로세스에서 단 한 번 전송하며 공급자 오류 본문은 넘기지 않는다."""
    client = None
    try:
        client = OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)
        result = _request(client, Path(path), model, max_chars)
        sender.send((None, result))
    except SpeechFailure as error:
        sender.send((error.code, None))
    except Exception:
        sender.send(("STT_UNAVAILABLE", None))
    finally:
        sender.close()
        if client is not None:
            client.close()


def _request(client, path: Path, model: str, max_chars: int) -> str:
    """실제 SDK 요청과 안전한 오류 코드 변환을 수행한다."""
    language_options = (
        {"extra_body": {"languages": ["ko"]}}
        if model == "gpt-transcribe" else {"language": "ko"}
    )
    try:
        with path.open("rb") as audio:
            result = client.audio.transcriptions.create(
                model=model, file=audio, response_format="json", **language_options
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
    if len(text) > max_chars:
        raise SpeechFailure("TRANSCRIPT_TOO_LONG")
    return text
