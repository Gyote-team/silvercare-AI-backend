"""Spring 내부 인증을 통과한 허용 기록 요약만 처리하는 API다."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.internal.v1.ai_result_document_router import require_internal_token
from app.application.use_cases.generate_previsit_summary import GeneratePrevisitSummaryUseCase
from app.contracts.previsit_summary import PrevisitSummaryRequest, PrevisitSummaryResponse
from app.core.config import Settings, get_settings
from app.infrastructure.providers.llm.openai_previsit_summarizer import OpenAiPrevisitSummarizer

router = APIRouter(prefix="/internal/previsit-summaries", tags=["Previsit summary"])


@router.post("/generate", response_model=PrevisitSummaryResponse)
def generate_previsit_summary(
    body: PrevisitSummaryRequest,
    _: Annotated[None, Depends(require_internal_token)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> PrevisitSummaryResponse:
    """전용 키 클라이언트로 허용된 기록만 요약한다."""
    return GeneratePrevisitSummaryUseCase(OpenAiPrevisitSummarizer(settings)).execute(body)
