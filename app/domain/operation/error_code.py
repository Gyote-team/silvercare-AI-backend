"""노션 FC-15 공통 오류 코드. 문서 분석·설명·채팅이 함께 쓴다."""

from enum import StrEnum


class AnalysisErrorCode(StrEnum):
    """실패 원인. 노션 FC-15 errorCode 중 지금 쓰는 값이다."""

    OBJECT_FETCH_FAILED = "OBJECT_FETCH_FAILED"
    DOCUMENT_PAGE_SPLIT_FAILED = "DOCUMENT_PAGE_SPLIT_FAILED"
    VLM_TIMEOUT = "VLM_TIMEOUT"
    VLM_RATE_LIMIT = "VLM_RATE_LIMIT"
    STRUCTURE_VALIDATION_FAILED = "STRUCTURE_VALIDATION_FAILED"
    AI_GENERATION_FAILED = "AI_GENERATION_FAILED"


class FailedStep(StrEnum):
    """실패한 단계. 노션 FC-15 failedStep 중 지금 쓰는 값이다.

    CHUNKING은 청킹 단계를 만들 때 추가한다.
    """

    OBJECT_FETCH = "OBJECT_FETCH"
    PAGE_SPLIT = "PAGE_SPLIT"
    VLM_EXTRACTION = "VLM_EXTRACTION"
    STRUCTURE_VALIDATION = "STRUCTURE_VALIDATION"
