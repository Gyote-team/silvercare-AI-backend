"""문서 분석의 상태 값과 실패 기록 형식을 정의한다."""

from dataclasses import dataclass
from enum import StrEnum


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


class AnalysisErrorCode(StrEnum):
    """분석 실패 원인. 노션 FC-15 errorCode 중 문서 분석이 쓰는 값이다."""

    OBJECT_FETCH_FAILED = "OBJECT_FETCH_FAILED"
    VLM_TIMEOUT = "VLM_TIMEOUT"
    VLM_RATE_LIMIT = "VLM_RATE_LIMIT"
    STRUCTURE_VALIDATION_FAILED = "STRUCTURE_VALIDATION_FAILED"
    AI_GENERATION_FAILED = "AI_GENERATION_FAILED"


class FailedStep(StrEnum):
    """노션 FC-15 DOCUMENT_ANALYSIS의 failedStep 중 지금 쓰는 값이다.

    CHUNKING은 청킹 단계를 만들 때 추가한다.
    """

    OBJECT_FETCH = "OBJECT_FETCH"
    VLM_EXTRACTION = "VLM_EXTRACTION"
    STRUCTURE_VALIDATION = "STRUCTURE_VALIDATION"


@dataclass(frozen=True)
class AnalysisFailure:
    """document_analyses에 남기는 실패 기록.

    detail_code는 error_message에 넣는 세부 코드 이름이다. 원문, 경로, 키는 담지 않는다.
    """

    error_code: AnalysisErrorCode
    failed_step: FailedStep
    retryable: bool
    detail_code: str
