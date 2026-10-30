"""FastAPI 앱을 만들고 상태 확인 API를 등록한다."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.error_handlers import register_error_handlers
from app.api.internal.v1.ai_result_document_router import router as ai_result_document_router
from app.api.internal.v1.document_analysis_router import router as document_analysis_router
from app.application.use_cases.generate_result_document import GenerateResultDocumentUseCase
from app.core.config import get_settings
from app.infrastructure.providers.llm.openai_compatible_client import OpenAiCompatibleLlmClient
from app.workers.document_analysis_task import resume_unfinished_analyses


@asynccontextmanager
async def lifespan(app: FastAPI):
    """LLM 클라이언트와 문서 분석 실행기를 앱 인스턴스마다 한 번 만들고 종료 시 닫는다."""
    settings = get_settings()
    client = OpenAiCompatibleLlmClient(settings)
    app.state.ai_result_document_use_case = GenerateResultDocumentUseCase(client)
    # 문서 분석은 API 스레드풀과 나눠 전용 스레드에서 돌린다.
    executor = ThreadPoolExecutor(
        max_workers=settings.document_analysis_max_workers,
        thread_name_prefix="document-analysis",
    )
    app.state.document_analysis_executor = executor
    if settings.document_analysis_resume_on_startup:
        # DB가 느리거나 죽어도 서버는 바로 뜨도록 조회도 실행기에서 한다.
        executor.submit(resume_unfinished_analyses, executor)
    try:
        yield
    finally:
        # 대기 중인 분석은 취소돼 PENDING으로 남고 다음 시작 때 다시 실행된다.
        # 실행 중인 분석은 끝날 때까지 프로세스 종료를 붙잡는다.
        executor.shutdown(wait=False, cancel_futures=True)
        client.close()


app = FastAPI(title="SilverCare AI Backend", lifespan=lifespan)
register_error_handlers(app)
app.include_router(ai_result_document_router)
app.include_router(document_analysis_router)


@app.get("/health")
def health() -> dict[str, str]:
    """서버가 살아 있는지 확인한다."""
    return {"status": "ok"}
