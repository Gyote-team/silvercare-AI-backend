"""자유 질문을 안전하게 분류하고 검색 전략을 결정한다."""

from app.contracts.ai_result_document import (
    ClassifyQuestionRequest,
    ClassifyQuestionResponse,
    QuestionType,
)
from app.domain.ai_result.question_classifier import QuestionClassifier
from app.domain.ai_result.safety_check import SafetyCheck


class ClassifyQuestionUseCase:
    def __init__(self) -> None:
        self._classifier = QuestionClassifier()
        self._safety = SafetyCheck()

    def classify(self, request: ClassifyQuestionRequest) -> ClassifyQuestionResponse:
        blocked = self._safety.classify(request.question)
        if blocked:
            return ClassifyQuestionResponse(
                request_id=request.request_id,
                question_type=blocked,
                confidence=1.0,
                classification_source="SAFETY",
                retrieval_strategy="NONE",
                blocked=True,
                block_reason="MEDICAL_DECISION_REQUEST",
            )

        question_type = self._classifier.classify(request.question, request.question_type_hint)
        if request.question_type_hint and question_type is not QuestionType.UNSUPPORTED:
            source, confidence = "HINT", 1.0
        elif question_type is not QuestionType.UNSUPPORTED:
            source, confidence = "RULE", 0.9
        else:
            # 짧은 지시대명사 질문은 최근 대화의 마지막 사용자 질문과 함께 규칙을 한 번 더 적용한다.
            previous = next(
                (m.content for m in reversed(request.conversation_context) if m.role == "USER"), None
            )
            contextual = self._classifier.classify(f"{previous or ''} {request.question}", None)
            if contextual is not QuestionType.UNSUPPORTED:
                question_type, source, confidence = contextual, "RULE", 0.75
            else:
                return ClassifyQuestionResponse(
                    request_id=request.request_id,
                    question_type=QuestionType.UNSUPPORTED,
                    confidence=0.4,
                    classification_source="RULE",
                    retrieval_strategy="NONE",
                    needs_clarification=True,
                    clarification_question="어떤 검사 결과나 문서 내용을 확인하고 싶은지 조금 더 구체적으로 알려 주세요.",
                )

        return ClassifyQuestionResponse(
            request_id=request.request_id,
            question_type=question_type,
            confidence=confidence,
            classification_source=source,
            retrieval_strategy=self._classifier.retrieval_strategy(question_type),
        )
