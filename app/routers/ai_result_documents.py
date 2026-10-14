"""Spring 내부 호출용 결과 문서 생성 API다."""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.core.config import Settings, get_settings
from app.infrastructure.providers.llm.openai_compatible_client import OpenAiCompatibleLlmClient
from app.schemas.ai_chat import GenerateResultDocumentRequest, GenerateResultDocumentResponse
from app.services.result_document_service import ResultDocumentService

router = APIRouter(prefix="/internal/ai-result-documents", tags=["AI result documents"])
SettingsDependency = Annotated[Settings, Depends(get_settings)]


def require_internal_token(
    settings: SettingsDependency,
    x_internal_token: Annotated[str | None, Header()] = None,
) -> None:
    """외부에서 AI 생성 endpoint를 직접 호출하지 못하게 한다."""
    if not x_internal_token or x_internal_token != settings.ai_internal_token.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid internal token",
        )


@router.post("/generate", response_model=GenerateResultDocumentResponse)
def generate(
    request: GenerateResultDocumentRequest,
    _: Annotated[None, Depends(require_internal_token)],
    settings: SettingsDependency,
) -> GenerateResultDocumentResponse:
    """근거가 검증된 상세 설명 또는 비교 문서를 생성한다."""
    service = ResultDocumentService(OpenAiCompatibleLlmClient(settings))
    return service.generate(request)
