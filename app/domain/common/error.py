"""Spring 백엔드와 맞춘 공통 HTTP 오류 계약이다."""

from dataclasses import dataclass
from enum import StrEnum


class CommonErrorCode(StrEnum):
    """공통 API 오류 코드다."""

    INVALID_REQUEST = "GLOBAL_002"
    INTERNAL_SERVER_ERROR = "GLOBAL_003"
    CONFLICT = "GLOBAL_004"
    UNAUTHORIZED = "AUTH_001"


_STATUS_BY_CODE: dict[CommonErrorCode, int] = {
    CommonErrorCode.INVALID_REQUEST: 400,
    CommonErrorCode.INTERNAL_SERVER_ERROR: 500,
    CommonErrorCode.CONFLICT: 409,
    CommonErrorCode.UNAUTHORIZED: 401,
}

_MESSAGE_BY_CODE: dict[CommonErrorCode, str] = {
    CommonErrorCode.INVALID_REQUEST: "요청 값이 올바르지 않습니다.",
    CommonErrorCode.INTERNAL_SERVER_ERROR: "서버 오류가 발생했습니다.",
    CommonErrorCode.CONFLICT: "다른 요청으로 자료가 변경되었습니다. 다시 조회한 후 시도해 주세요.",
    CommonErrorCode.UNAUTHORIZED: "인증 정보가 올바르지 않습니다.",
}


@dataclass(frozen=True)
class ApiError(Exception):
    """공통 코드와 안전한 사용자 메시지만 HTTP 계층으로 전달한다."""

    code: CommonErrorCode

    def __post_init__(self) -> None:
        Exception.__init__(self, self.code.value)

    @property
    def status_code(self) -> int:
        """공통 계약에 맞는 HTTP 상태 코드를 반환한다."""
        return _STATUS_BY_CODE[self.code]

    @property
    def message(self) -> str:
        """의료 원문이나 내부 정보를 포함하지 않는 사용자 메시지를 반환한다."""
        return _MESSAGE_BY_CODE[self.code]
