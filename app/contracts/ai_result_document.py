"""근거 기반 의료기록 채팅의 내부 API 계약이다."""

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.alias_generators import to_camel


class QuestionType(StrEnum):
    """가이드형 채팅에서 지원하는 질문 유형이다."""

    VISIT_CHANGE_REPORT = "VISIT_CHANGE_REPORT"
    TEST_RESULT_COMPARISON = "TEST_RESULT_COMPARISON"
    LAB_RESULT_COMPARISON = "LAB_RESULT_COMPARISON"  # legacy request alias
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

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    citation_id: UUID
    source_type: str = Field(pattern="^(DOCUMENT_CHUNK|EXTRACTED_ITEM|HEALTH_RECORD)$")
    source_id: UUID | None = None
    document_id: UUID | None = None
    health_record_id: UUID | None = None
    page_no: int | None = Field(default=None, ge=1)
    source_text: str = Field(min_length=1, max_length=6000)
    label: str | None = Field(default=None, max_length=200)
    value: str | None = Field(default=None, max_length=500)
    unit: str | None = Field(default=None, max_length=50)
    reference_range: str | None = Field(default=None, max_length=200)
    recorded_at: str | None = Field(default=None, max_length=40)

    @model_validator(mode="after")
    def normalize_source_identifier(self):
        """Spring의 공통 sourceId를 근거 유형별 식별자로 정규화한다."""
        if self.source_type == "HEALTH_RECORD":
            self.health_record_id = self.health_record_id or self.source_id
            if self.health_record_id is None:
                raise ValueError(
                    "HEALTH_RECORD 근거에는 healthRecordId 또는 sourceId가 필요합니다."
                )
        else:
            self.document_id = self.document_id or self.source_id
            if self.document_id is None:
                raise ValueError("문서 근거에는 documentId 또는 sourceId가 필요합니다.")
        return self


class GenerateResultDocumentRequest(BaseModel):
    """Spring이 권한 확인 뒤 AI 서버로 보내는 작업 요청이다."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    request_id: UUID
    patient_id: UUID
    actor_user_id: UUID
    question: str = Field(min_length=1, max_length=2000)
    question_type_hint: QuestionType | None = None
    allowed_visit_ids: list[UUID] = Field(default_factory=list)
    allowed_document_ids: list[UUID] = Field(default_factory=list)
    allowed_health_record_ids: list[UUID] = Field(default_factory=list)
    evidence: list[EvidenceSource] = Field(default_factory=list, max_length=40)
    detailed_document: bool = False

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("질문은 공백만으로 구성할 수 없습니다.")
        return normalized

    @model_validator(mode="after")
    def validate_evidence_scope(self):
        """모든 근거가 Spring이 허용한 리소스 범위 안에 있는지 검증한다."""
        allowed_documents = set(self.allowed_document_ids)
        allowed_health_records = set(self.allowed_health_record_ids)
        for source in self.evidence:
            if source.source_type == "HEALTH_RECORD":
                if source.health_record_id not in allowed_health_records:
                    raise ValueError("허용 범위 밖의 건강기록 근거가 포함되어 있습니다.")
            elif source.document_id not in allowed_documents:
                raise ValueError("허용 범위 밖의 문서 근거가 포함되어 있습니다.")
        return self


class AnswerSentence(BaseModel):
    """화면에 노출 가능한 문장과 근거의 연결이다."""

    text: str = Field(min_length=1, max_length=2000)
    citation_ids: list[UUID] = Field(max_length=8)


class AnswerItem(BaseModel):
    kind: str = Field(pattern="^(PARAGRAPH|BULLET|CALLOUT)$")
    text: str = Field(min_length=1, max_length=2000)
    citation_ids: list[UUID] = Field(max_length=8)


class AnswerSection(BaseModel):
    heading: str = Field(min_length=1, max_length=200)
    items: list[AnswerItem] = Field(min_length=1, max_length=12)


class GeneratedAnswer(BaseModel):
    """GPT가 반환해야 하는 제한된 JSON 형식이다."""

    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=2000)
    summary_citation_ids: list[UUID] = Field(default_factory=list, max_length=8)
    sections: list[AnswerSection] = Field(min_length=1, max_length=8)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_test_fixture(cls, value):
        """기존 테스트 픽스처만 새 구조로 정규화하며 공개 스키마는 새 계약으로 유지한다."""
        if isinstance(value, dict) and "sections" not in value and "sentences" in value:
            sentences = value.get("sentences", [])
            summary = value.get("answer", "답변")
            matching_summary = next(
                (
                    item
                    for item in sentences
                    if (item.text if hasattr(item, "text") else item.get("text")) == summary
                ),
                None,
            )
            summary_citation_ids = (
                matching_summary.citation_ids
                if hasattr(matching_summary, "citation_ids")
                else matching_summary.get("citation_ids", [])
                if matching_summary
                else []
            )
            return {
                "title": value.get("title", "답변"),
                "summary": summary,
                "summary_citation_ids": summary_citation_ids,
                "sections": [
                    {
                        "heading": "상세 내용",
                        "items": [
                            {
                                "kind": "PARAGRAPH",
                                "text": item.text if hasattr(item, "text") else item.get("text"),
                                "citation_ids": item.citation_ids
                                if hasattr(item, "citation_ids")
                                else item.get("citation_ids", []),
                            }
                            for item in sentences
                        ],
                    }
                ],
            }
        return value


class LlmUsage(BaseModel):
    model: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class ConversationMessage(BaseModel):
    role: str = Field(pattern="^(USER|ASSISTANT)$")
    content: str = Field(min_length=1, max_length=2000)


class ClassifyQuestionRequest(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    request_id: UUID
    question: str = Field(min_length=1, max_length=2000)
    question_type_hint: QuestionType | None = None
    conversation_context: list[ConversationMessage] = Field(default_factory=list, max_length=6)


class ClassifyQuestionResponse(BaseModel):
    request_id: UUID
    question_type: QuestionType
    confidence: float = Field(ge=0, le=1)
    classification_source: str = Field(pattern="^(HINT|RULE|LLM|SAFETY)$")
    retrieval_strategy: str
    needs_clarification: bool = False
    clarification_question: str | None = None
    blocked: bool = False
    block_reason: str | None = None
    policy_version: str = "safety-v2"
    usage: LlmUsage | None = None


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
    content_format: str = "MARKDOWN"
    prompt_version: str = "chat-grounded-v2"
    sentences: list[AnswerSentence] = Field(default_factory=list)
    sources: list[EvidenceSource] = Field(default_factory=list)
    usage: LlmUsage | None = None
