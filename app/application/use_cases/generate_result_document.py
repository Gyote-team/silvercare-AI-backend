"""근거 기반 상세 설명·비교 문서 생성 유스케이스다."""

from uuid import uuid4

from app.application.markdown_renderer import render_markdown
from app.application.ports.chunk_evidence_retriever import ChunkEvidenceRetriever
from app.application.ports.text_embedder import EmbeddingError, TextEmbedder
from app.application.prompt_builder import PROMPT_VERSION, PromptBuilder
from app.contracts.ai_result_document import (
    AnswerSection,
    AnswerSentence,
    EvidenceSource,
    GeneratedAnswer,
    GenerateResultDocumentRequest,
    GenerateResultDocumentResponse,
    QuestionType,
)
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
        chunk_retriever: ChunkEvidenceRetriever | None = None,
        text_embedder: TextEmbedder | None = None,
        retrieval_enabled: bool = False,
        retrieval_top_k: int = 8,
        retrieval_min_similarity: float = 0.30,
    ) -> None:
        self._llm_client = llm_client
        self._safety_checks = safety_checks or SafetyCheck()
        self._classifier = classifier or QuestionClassifier()
        self._prompts = prompt_builder or PromptBuilder()
        self._chunk_retriever = chunk_retriever
        self._text_embedder = text_embedder
        self._retrieval_enabled = retrieval_enabled
        self._retrieval_top_k = retrieval_top_k
        self._retrieval_min_similarity = retrieval_min_similarity

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
        enriched, retrieval_error = self._with_retrieved_chunks(request, question_type)
        if retrieval_error is not None:
            return retrieval_error
        if not enriched.evidence:
            return GenerateResultDocumentResponse(
                request_id=request.request_id,
                question_type=question_type,
                blocked=False,
                result_status="INSUFFICIENT",
                user_message="답변에 사용할 수 있는 의료문서 또는 건강기록 근거를 찾지 못했어요.",
            )

        try:
            generated = self._llm_client.generate_grounded_answer(
                self._prompts.system(question_type, enriched.detailed_document),
                self._prompts.user(enriched, question_type),
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
        return self._validated_response(enriched, question_type, answer, usage)

    def _with_retrieved_chunks(
        self, request: GenerateResultDocumentRequest, question_type: QuestionType
    ) -> tuple[GenerateResultDocumentRequest, GenerateResultDocumentResponse | None]:
        """질문 검색 청크를 기존 구조화 근거 앞에 합쳐 반환한다."""
        structured = list(request.evidence[:24])
        if not self._retrieval_enabled or not request.allowed_document_ids:
            return request.model_copy(update={"evidence": structured}), None
        if self._chunk_retriever is None or self._text_embedder is None:
            return request, self._retrieval_failure(
                request, question_type, "RETRIEVAL_NOT_CONFIGURED", retryable=False
            )
        try:
            [query_vector] = self._text_embedder.embed([request.question])
            chunks = self._chunk_retriever.retrieve(
                patient_id=request.patient_id,
                allowed_document_ids=request.allowed_document_ids,
                query_vector=query_vector,
                model_name=self._text_embedder.model_name,
                top_k=self._retrieval_top_k,
                min_similarity=self._retrieval_min_similarity,
            )
        except EmbeddingError as error:
            retryable = error.code.value in {
                "TIMEOUT",
                "RATE_LIMITED",
                "API_ERROR",
                "CONNECTION_FAILED",
            }
            return request, self._retrieval_failure(
                request, question_type, f"EMBEDDING_{error.code.value}", retryable=retryable
            )
        except PermissionError:
            return request, self._retrieval_failure(
                request, question_type, "RETRIEVAL_SCOPE_VIOLATION", retryable=False
            )
        except Exception:
            return request, self._retrieval_failure(
                request, question_type, "RETRIEVAL_FAILED", retryable=True
            )

        allowed = set(request.allowed_document_ids)
        if any(chunk.document_id not in allowed for chunk in chunks):
            return request, self._retrieval_failure(
                request, question_type, "RETRIEVAL_SCOPE_VIOLATION", retryable=False
            )
        retrieved = [
            EvidenceSource(
                citation_id=uuid4(),
                source_type="DOCUMENT_CHUNK",
                source_id=chunk.chunk_id,
                document_id=chunk.document_id,
                page_no=chunk.page_no,
                source_text=chunk.content,
            )
            for chunk in chunks
        ]
        merged: list[EvidenceSource] = []
        seen: set[tuple[object, object, str]] = set()
        for source in [*retrieved, *structured]:
            key = (source.document_id, source.page_no, source.source_text.strip())
            if key not in seen:
                seen.add(key)
                merged.append(source)
        return request.model_copy(update={"evidence": merged}), None

    @staticmethod
    def _retrieval_failure(
        request: GenerateResultDocumentRequest,
        question_type: QuestionType,
        error_code: str,
        *,
        retryable: bool,
    ) -> GenerateResultDocumentResponse:
        return GenerateResultDocumentResponse(
            request_id=request.request_id,
            question_type=question_type,
            blocked=False,
            failed=True,
            error_code=error_code,
            retryable=retryable,
            user_message="문서 근거를 검색하지 못했어요. 잠시 후 다시 시도해 주세요.",
        )

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
        summary_is_valid = bool(answer.summary_citation_ids) and set(
            answer.summary_citation_ids
        ).issubset(allowed)
        original_count = 1
        if summary_is_valid:
            sentences.append(
                AnswerSentence(text=answer.summary, citation_ids=answer.summary_citation_ids)
            )
        for section in answer.sections:
            original_count += len(section.items)
            valid_items = [
                item
                for item in section.items
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
                usage=usage,
                user_message="생성된 답변에서 원문 근거를 확인하지 못했어요.",
            )
        result_status = "COMPLETE" if len(sentences) == original_count else "PARTIAL"
        used_citations = {
            citation_id for sentence in sentences for citation_id in sentence.citation_ids
        }
        used_sources = [
            source for source in request.evidence if source.citation_id in used_citations
        ]
        prompt_version = f"{PROMPT_VERSION}-detail" if request.detailed_document else PROMPT_VERSION
        return GenerateResultDocumentResponse(
            request_id=request.request_id,
            question_type=question_type,
            blocked=False,
            result_status=result_status,
            title=answer.title,
            answer=render_markdown(
                answer.title, answer.summary if summary_is_valid else "", sections
            ),
            prompt_version=prompt_version,
            sentences=sentences,
            sources=used_sources,
            usage=usage,
        )
