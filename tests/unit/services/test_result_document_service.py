from uuid import uuid4

from app.schemas.ai_chat import (
    AnswerSentence,
    EvidenceSource,
    GeneratedAnswer,
    GenerateResultDocumentRequest,
    QuestionType,
)
from app.services.result_document_service import ResultDocumentService


class FakeLlm:
    def __init__(self, answer: GeneratedAnswer) -> None:
        self.answer = answer

    def generate_grounded_answer(self, system_prompt: str, user_prompt: str) -> GeneratedAnswer:
        return self.answer


def request_with_evidence(citation_id):
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
    service = ResultDocumentService(
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
    allowed = uuid4()
    service = ResultDocumentService(
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
    allowed = uuid4()
    service = ResultDocumentService(
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
