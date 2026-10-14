"""근거 기반 AI 결과 문서 생성 오류를 표현한다."""

from enum import StrEnum


class AiResultErrorCode(StrEnum):
    """결과 문서 생성 중 Spring에 전달할 수 있는 안전한 오류 코드다."""

    AI_TIMEOUT = "AI_TIMEOUT"
    AI_RATE_LIMIT = "AI_RATE_LIMIT"
    AI_GENERATION_FAILED = "AI_GENERATION_FAILED"
    AI_RESPONSE_INVALID = "AI_RESPONSE_INVALID"


class AiResultGenerationError(Exception):
    """외부 LLM 실패를 코드·재시도 가능 여부로 정규화한다."""

    def __init__(self, code: AiResultErrorCode, *, retryable: bool) -> None:
        super().__init__(code.value)
        self.code = code
        self.retryable = retryable

    def user_message(self) -> str:
        """민감한 공급자 세부 정보를 제외한 사용자 안내 문구를 반환한다."""
        if self.code is AiResultErrorCode.AI_TIMEOUT:
            return "AI 응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요."
        if self.code is AiResultErrorCode.AI_RATE_LIMIT:
            return "AI 요청이 일시적으로 많습니다. 잠시 후 다시 시도해 주세요."
        if self.retryable:
            return "AI 처리 중 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
        return "AI 응답을 안전하게 처리하지 못했습니다. 문서와 질문을 확인한 뒤 다시 시도해 주세요."
