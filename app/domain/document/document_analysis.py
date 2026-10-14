"""문서 분석의 상태 값과 실패 기록 형식을 정의한다."""

from dataclasses import dataclass
from enum import StrEnum

from app.domain.operation.error_code import AnalysisErrorCode, FailedStep


class DocumentStatus(StrEnum):
    """문서 상태. documents.status의 CHECK 제약 값과 같다."""

    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    READY = "READY"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    FAILED = "FAILED"
    DELETED = "DELETED"


class AnalysisStatus(StrEnum):
    """분석 작업 상태. document_analyses.status의 CHECK 제약 값과 같다."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class AnalysisFailure:
    """document_analyses에 남기는 실패 기록.

    detail_code는 error_message에 넣는 세부 코드 이름이다. 원문, 경로, 키는 담지 않는다.
    """

    error_code: AnalysisErrorCode
    failed_step: FailedStep
    retryable: bool
    detail_code: str
