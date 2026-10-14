from uuid import uuid4

from app.application.use_cases.generate_result_document import GenerateResultDocumentUseCase
from app.contracts.ai_result_document import (
    AnswerSentence,
    EvidenceSource,
    GeneratedAnswer,
    GenerateResultDocumentRequest,
    QuestionType,
)
from app.domain.ai_result.error import AiResultErrorCode, AiResultGenerationError


class FakeLlm:
    """검증 가능한 고정 LLM 응답을 반환한다."""

    def __init__(self, answer: GeneratedAnswer) -> None:
        self.answer = answer

    def generate_grounded_answer(self, system_prompt: str, user_prompt: str) -> GeneratedAnswer:
        """테스트용 생성 결과를 반환한다."""
        return self.answer


class FailingLlm:
    """정규화된 LLM 생성 오류를 발생시킨다."""

    def generate_grounded_answer(self, system_prompt: str, user_prompt: str) -> GeneratedAnswer:
        """AI 공급자 시간 초과를 흉내 낸다."""
        raise AiResultGenerationError(AiResultErrorCode.AI_TIMEOUT, retryable=True)


def request_with_evidence(citation_id):
    """원문 근거가 하나인 결과 문서 요청을 만든다."""
    return GenerateResultDocumentRequest(
        request_id=uuid4(),
        patient_id=uuid4(),
        actor_user_id=uuid4(),
        question="이 문서를 자세하게 쉽게 설명해 줘",
        evidence=[
            EvidenceSource(
                citation_id=citation_id,
                source_type="DOCUMENT_CHUNK",
                source_text="HbA1c 6.5 %",
            )
        ],
    )


def test_medical_decision_question_is_blocked_without_llm_call():
    """진단·처방 판단 질문은 LLM 호출 전에 차단한다."""
    service = GenerateResultDocumentUseCase(
        FakeLlm(
            GeneratedAnswer(
                title="unused",
                answer="unused",
                sentences=[AnswerSentence(text="unused", citation_ids=[uuid4()])],
            )
        )
    )
    request = request_with_evidence(uuid4()).model_copy(update={"question": "약을 끊어도 돼?"})

    result = service.generate(request)

    assert result.blocked is True
    assert result.question_type is QuestionType.MEDICAL_DECISION_REQUEST


def test_response_without_allowed_citation_is_not_exposed():
    """허용되지 않은 citation만 든 문장은 응답에서 제외한다."""
    allowed = uuid4()
    service = GenerateResultDocumentUseCase(
        FakeLlm(
            GeneratedAnswer(
                title="설명",
                answer="근거 없는 답변",
                sentences=[AnswerSentence(text="근거 없는 문장", citation_ids=[uuid4()])],
            )
        )
    )

    result = service.generate(request_with_evidence(allowed))

    assert result.result_status == "INSUFFICIENT"
    assert result.sentences == []


def test_grounded_response_is_returned():
    """허용 근거를 인용한 문장은 COMPLETE 결과로 반환한다."""
    allowed = uuid4()
    service = GenerateResultDocumentUseCase(
        FakeLlm(
            GeneratedAnswer(
                title="문서 쉬운 설명",
                answer="검사 결과에 HbA1c 6.5 %가 적혀 있습니다.",
                sentences=[
                    AnswerSentence(
                        text="HbA1c 6.5 %가 적혀 있습니다.",
                        citation_ids=[allowed],
                    )
                ],
            )
        )
    )

    result = service.generate(request_with_evidence(allowed))

    assert result.result_status == "COMPLETE"
    assert result.title == "문서 쉬운 설명"


def test_llm_timeout_is_returned_as_retryable_failed_result():
    """LLM 시간 초과는 민감한 세부 정보 없이 재시도 가능한 실패로 반환한다."""
    result = GenerateResultDocumentUseCase(FailingLlm()).generate(request_with_evidence(uuid4()))

    assert result.failed is True
    assert result.error_code == "AI_TIMEOUT"
    assert result.retryable is True
    assert result.answer is None
