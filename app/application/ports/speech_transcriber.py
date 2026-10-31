"""음성 인식 공급자와 작업자를 분리한다."""

from pathlib import Path
from typing import Protocol


class SpeechFailure(Exception):
    """원문·인증정보가 포함되지 않은 안전한 실패 코드만 보관한다."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class SpeechTranscriber(Protocol):
    """원본 음성을 초안으로만 변환한다."""

    def transcribe(self, path: Path) -> str:
        """사용자 확정이나 진료 판단 없이 인식된 텍스트를 반환한다."""
