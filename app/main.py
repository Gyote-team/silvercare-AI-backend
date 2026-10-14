"""FastAPI 앱을 만들고 상태 확인 API를 등록한다."""

from fastapi import FastAPI

from app.api.error_handlers import register_error_handlers
from app.api.internal.v1.ai_result_document_router import router as ai_result_document_router

app = FastAPI(title="SilverCare AI Backend")
register_error_handlers(app)
app.include_router(ai_result_document_router)


@app.get("/health")
def health() -> dict[str, str]:
    """서버가 살아 있는지 확인한다."""
    return {"status": "ok"}
