"""근거 기반 상세 설명·비교 문서 생성 유스케이스다."""

import json

from app.infrastructure.providers.llm.openai_compatible_client import LlmClient
from app.schemas.ai_chat import (
    GeneratedAnswer,
    GenerateResultDocumentRequest,
    GenerateResultDocumentResponse,
    QuestionType,
)
from app.services.question_classifier_service import QuestionClassifierService
from app.services.safety_check_service import SafetyCheckService


class ResultDocumentService:
    """LLM 호출 전후에 안전성과 citation 무결성을 강제한다."""

    def __init__(
        self,
        llm_client: LlmClient,
        safety_checks: SafetyCheckService | None = None,
        classifier: QuestionClassifierService | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._safety_checks = safety_checks or SafetyCheckService()
        self._classifier = classifier or QuestionClassifierService()

    def generate(self, request: GenerateResultDocumentRequest) -> GenerateResultDocumentResponse:
        blocked_type = self._safety_checks.classify(request.question)
        if blocked_type:
            return GenerateResultDocumentResponse(
                request_id=request.request_id,
                question_type=blocked_type,
                blocked=True,
                block_reason="MEDICAL_DECISION_REQUEST",
                user_message=self._safety_checks.user_message(),
            )

        question_type = self._classifier.classify(request.question, request.question_type_hint)
        if question_type is QuestionType.UNSUPPORTED:
            return GenerateResultDocumentResponse(
                request_id=request.request_id,
                question_type=question_type,
                blocked=True,
                block_reason="UNSUPPORTED_QUESTION",
                user_message=(
                    "검사 결과 비교, 문서 설명, 복약 정보, 일정 확인처럼 "
                    "의료기록에서 확인 가능한 질문으로 바꿔 주세요."
                ),
            )
        if not request.evidence:
            return GenerateResultDocumentResponse(
                request_id=request.request_id,
                question_type=question_type,
                blocked=False,
                result_status="INSUFFICIENT",
                user_message="답변에 사용할 수 있는 의료문서 또는 건강기록 근거를 찾지 못했어요.",
            )

        answer = self._llm_client.generate_grounded_answer(
            self._system_prompt(question_type),
            self._user_prompt(request, question_type),
        )
        return self._validated_response(request, question_type, answer)

    @staticmethod
    def _system_prompt(question_type: QuestionType) -> str:
        return f"""당신은 SilverCare의 의료기록 설명 도우미입니다.
질문 유형은 {question_type.value}입니다.
제공된 근거에 없는 사실, 진단 확정, 약 변경·중단, 치료 결정을 절대 생성하지 마세요.
수치·단위·날짜는 제공된 값 그대로 사용하세요.
답변의 모든 문장은 citation_ids로 제공된 근거를 하나 이상 인용해야 합니다.
반드시 title, answer, sentences 배열을 가진 JSON만 반환하세요.
sentences 원소는 text와 citation_ids 배열을 가져야 합니다."""

    @staticmethod
    def _user_prompt(request: GenerateResultDocumentRequest, question_type: QuestionType) -> str:
        evidence = [source.model_dump(mode="json") for source in request.evidence]
        return json.dumps(
            {
                "question": request.question,
                "questionType": question_type.value,
                "evidence": evidence,
                "instruction": (
                    "근거를 충분히 활용해 상세하게 설명하되, 문서에 적힌 내용과 "
                    "계산 가능한 비교만 다루세요."
                ),
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _validated_response(
        request: GenerateResultDocumentRequest,
        question_type: QuestionType,
        answer: GeneratedAnswer,
    ) -> GenerateResultDocumentResponse:
        allowed = {source.citation_id for source in request.evidence}
        sentences = [
            sentence
            for sentence in answer.sentences
            if set(sentence.citation_ids).issubset(allowed)
        ]
        if not sentences:
            return GenerateResultDocumentResponse(
                request_id=request.request_id,
                question_type=question_type,
                blocked=False,
                result_status="INSUFFICIENT",
                user_message="생성된 답변에서 원문 근거를 확인하지 못했어요.",
            )
        result_status = "COMPLETE" if len(sentences) == len(answer.sentences) else "PARTIAL"
        return GenerateResultDocumentResponse(
            request_id=request.request_id,
            question_type=question_type,
            blocked=False,
            result_status=result_status,
            title=answer.title,
            answer="\n\n".join(sentence.text for sentence in sentences),
            sentences=sentences,
        )
