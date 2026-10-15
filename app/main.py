"""FastAPI 앱을 만들고 상태 확인 API를 등록한다."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.error_handlers import register_error_handlers
from app.api.internal.v1.ai_result_document_router import router as ai_result_document_router
from app.application.use_cases.generate_result_document import GenerateResultDocumentUseCase
from app.core.config import get_settings
from app.infrastructure.providers.llm.openai_compatible_client import OpenAiCompatibleLlmClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    """LLM 클라이언트를 앱 인스턴스마다 한 번 생성하고 종료 시 닫는다."""
    client = OpenAiCompatibleLlmClient(get_settings())
    app.state.ai_result_document_use_case = GenerateResultDocumentUseCase(client)
    try:
        yield
    finally:
        client.close()


app = FastAPI(title="SilverCare AI Backend", lifespan=lifespan)
register_error_handlers(app)
app.include_router(ai_result_document_router)


@app.get("/health")
def health() -> dict[str, str]:
    """서버가 살아 있는지 확인한다."""
    return {"status": "ok"}
