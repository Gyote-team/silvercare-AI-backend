"""공통 JSON 오류 응답을 FastAPI 예외에 적용한다."""

import logging
from datetime import UTC, datetime

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.domain.common.error import ApiError, CommonErrorCode

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    """입력·인증·예상 밖 오류를 Spring과 같은 JSON 구조로 등록한다."""

    @app.exception_handler(ApiError)
    async def handle_api_error(_: Request, error: ApiError) -> JSONResponse:
        """의도적으로 정의한 공통 오류를 코드·메시지·시각으로 반환한다."""
        return _error_response(error.status_code, error.code, error.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, __: RequestValidationError) -> JSONResponse:
        """본문·경로·쿼리 검증 실패를 GLOBAL_002로 통일한다."""
        return _error_response(
            400,
            CommonErrorCode.INVALID_REQUEST,
            "요청 값이 올바르지 않습니다.",
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
        """내부 예외 세부 정보 없이 GLOBAL_003만 반환한다."""
        logger.exception(
            "처리되지 않은 AI API 오류: method=%s path=%s type=%s",
            request.method,
            request.url.path,
            type(error).__name__,
        )
        return _error_response(
            500,
            CommonErrorCode.INTERNAL_SERVER_ERROR,
            "서버 오류가 발생했습니다.",
        )


def _error_response(status_code: int, code: CommonErrorCode, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "code": code.value,
            "message": message,
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )
