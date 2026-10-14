"""근거 기반 의료기록 채팅의 내부 API 계약이다."""

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class QuestionType(StrEnum):
    """가이드형 채팅에서 지원하는 질문 유형이다."""

    LAB_RESULT_COMPARISON = "LAB_RESULT_COMPARISON"
    LAB_REFERENCE_RANGE = "LAB_REFERENCE_RANGE"
    DOCUMENT_EASY_EXPLANATION = "DOCUMENT_EASY_EXPLANATION"
    MEDICATION_SUMMARY = "MEDICATION_SUMMARY"
    FOLLOW_UP_SCHEDULE = "FOLLOW_UP_SCHEDULE"
    VISIT_CHECKLIST = "VISIT_CHECKLIST"
    SOURCE_LOOKUP = "SOURCE_LOOKUP"
    MEDICAL_DECISION_REQUEST = "MEDICAL_DECISION_REQUEST"
    UNSUPPORTED = "UNSUPPORTED"


class EvidenceSource(BaseModel):
    """LLM에 전달할 수 있도록 권한이 검증된 원문 근거를 정규화한 값이다."""

    citation_id: UUID
    source_type: str = Field(pattern="^(DOCUMENT_CHUNK|EXTRACTED_ITEM|HEALTH_RECORD)$")
    document_id: UUID | None = None
    health_record_id: UUID | None = None
    page_no: int | None = Field(default=None, ge=1)
    source_text: str = Field(min_length=1, max_length=6000)
    label: str | None = Field(default=None, max_length=200)
    value: str | None = Field(default=None, max_length=500)
    unit: str | None = Field(default=None, max_length=50)
    reference_range: str | None = Field(default=None, max_length=200)
    recorded_at: str | None = Field(default=None, max_length=40)


class GenerateResultDocumentRequest(BaseModel):
    """Spring이 권한 확인 뒤 AI 서버로 보내는 작업 요청이다."""

    request_id: UUID
    patient_id: UUID
    actor_user_id: UUID
    question: str = Field(min_length=1, max_length=2000)
    question_type_hint: QuestionType | None = None
    allowed_visit_ids: list[UUID] = Field(default_factory=list)
    allowed_document_ids: list[UUID] = Field(default_factory=list)
    allowed_health_record_ids: list[UUID] = Field(default_factory=list)
    evidence: list[EvidenceSource] = Field(default_factory=list, max_length=40)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("질문은 공백만으로 구성할 수 없습니다.")
        return normalized


class AnswerSentence(BaseModel):
    """화면에 노출 가능한 문장과 근거의 연결이다."""

    text: str = Field(min_length=1, max_length=2000)
    citation_ids: list[UUID] = Field(min_length=1, max_length=8)


class GeneratedAnswer(BaseModel):
    """Gemini가 반환해야 하는 제한된 JSON 형식이다."""

    title: str = Field(min_length=1, max_length=200)
    answer: str = Field(min_length=1, max_length=12000)
    sentences: list[AnswerSentence] = Field(min_length=1, max_length=30)


class GenerateResultDocumentResponse(BaseModel):
    """AI 생성 결과 또는 안전 차단 결과다."""

    request_id: UUID
    question_type: QuestionType
    blocked: bool
    failed: bool = False
    block_reason: str | None = None
    error_code: str | None = None
    retryable: bool = False
    policy_version: str = "safety-v1"
    user_message: str | None = None
    result_status: str | None = None
    title: str | None = None
    answer: str | None = None
    sentences: list[AnswerSentence] = Field(default_factory=list)
