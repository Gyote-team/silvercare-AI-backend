"""페이지 PNG 한 장에서 추출 결과를 얻는 VLM 포트와 결과 형식, 예외를 정의한다."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from app.domain.document.page_extraction import PageExtraction
from app.domain.document.page_split import PageImage


class PageExtractionErrorCode(StrEnum):
    """페이지 추출이 실패한 이유."""

    MISSING_API_KEY = "MISSING_API_KEY"  # API 키가 설정되지 않음
    TIMEOUT = "TIMEOUT"  # 제한 시간 안에 응답이 없음
    RATE_LIMITED = "RATE_LIMITED"  # 호출 한도 초과(429)
    API_ERROR = "API_ERROR"  # provider 서버 오류(5xx)나 네트워크 오류
    REQUEST_REJECTED = "REQUEST_REJECTED"  # 인증 실패, 잘못된 요청 등 429 외 4xx
    INCOMPLETE = "INCOMPLETE"  # 출력 토큰 상한 등으로 응답이 끝나지 않음
    REFUSED = "REFUSED"  # 모델이 응답을 거절함
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"  # 응답이 출력 형식과 맞지 않음


class PageExtractionError(Exception):
    """페이지 추출 실패. 메시지에는 오류 코드만 담고 키, 이미지, 문서 원문은 담지 않는다."""

    def __init__(self, code: PageExtractionErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


@dataclass(frozen=True)
class VlmUsage:
    """VLM 호출 한 번에 쓴 모델, 프롬프트 버전, 토큰 수.

    reasoning_tokens는 output_tokens에 포함된다.
    """

    model_name: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int


@dataclass(frozen=True)
class PageExtractionResult:
    """페이지 한 장의 추출 결과와 그 호출의 사용량."""

    extraction: PageExtraction
    usage: VlmUsage


class PageExtractor(Protocol):
    """페이지 PNG 한 장을 VLM에 보내 추출 결과를 받는 인터페이스."""

    def extract_page(self, page: PageImage) -> PageExtractionResult:
        """추출 결과와 사용량을 반환한다. 실패하면 PageExtractionError를 던진다."""
        ...
