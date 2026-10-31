"""근거 기반 상세 설명·비교 문서 생성 유스케이스다."""

from app.contracts.ai_result_document import (
    AnswerSection,
    AnswerSentence,
    GeneratedAnswer,
    GenerateResultDocumentRequest,
    GenerateResultDocumentResponse,
    QuestionType,
)
from app.application.markdown_renderer import render_markdown
from app.application.prompt_builder import PROMPT_VERSION, PromptBuilder
from app.domain.ai_result.error import AiResultGenerationError
from app.domain.ai_result.question_classifier import QuestionClassifier
from app.domain.ai_result.safety_check import SafetyCheck
from app.infrastructure.providers.llm.openai_compatible_client import LlmClient


class GenerateResultDocumentUseCase:
    """LLM 호출 전후에 안전성과 citation 무결성을 강제한다."""

    def __init__(
        self,
        llm_client: LlmClient,
        safety_checks: SafetyCheck | None = None,
        classifier: QuestionClassifier | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._safety_checks = safety_checks or SafetyCheck()
        self._classifier = classifier or QuestionClassifier()
        self._prompts = prompt_builder or PromptBuilder()

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

        try:
            generated = self._llm_client.generate_grounded_answer(
                self._prompts.system(question_type), self._prompts.user(request, question_type)
            )
            answer, usage = generated if isinstance(generated, tuple) else (generated, None)
        except AiResultGenerationError as error:
            return GenerateResultDocumentResponse(
                request_id=request.request_id,
                question_type=question_type,
                blocked=False,
                failed=True,
                error_code=error.code.value,
                retryable=error.retryable,
                user_message=error.user_message(),
            )
        return self._validated_response(request, question_type, answer, usage)

    @staticmethod
    def _validated_response(
        request: GenerateResultDocumentRequest,
        question_type: QuestionType,
        answer: GeneratedAnswer,
        usage=None,
    ) -> GenerateResultDocumentResponse:
        allowed = {source.citation_id for source in request.evidence}
        sections: list[AnswerSection] = []
        sentences: list[AnswerSentence] = []
        original_count = 0
        for section in answer.sections:
            original_count += len(section.items)
            valid_items = [
                item for item in section.items
                if item.citation_ids and set(item.citation_ids).issubset(allowed)
            ]
            if valid_items:
                sections.append(AnswerSection(heading=section.heading, items=valid_items))
                sentences.extend(
                    AnswerSentence(text=item.text, citation_ids=item.citation_ids)
                    for item in valid_items
                )
        if not sentences:
            return GenerateResultDocumentResponse(
                request_id=request.request_id,
                question_type=question_type,
                blocked=False,
                result_status="INSUFFICIENT",
                user_message="생성된 답변에서 원문 근거를 확인하지 못했어요.",
            )
        result_status = "COMPLETE" if len(sentences) == original_count else "PARTIAL"
        return GenerateResultDocumentResponse(
            request_id=request.request_id,
            question_type=question_type,
            blocked=False,
            result_status=result_status,
            title=answer.title,
            answer=render_markdown(answer.title, answer.summary, sections),
            prompt_version=PROMPT_VERSION,
            sentences=sentences,
            usage=usage,
        )
