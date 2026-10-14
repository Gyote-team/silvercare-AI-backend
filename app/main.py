"""FastAPI 앱을 만들고 상태 확인 API를 등록한다."""

from fastapi import FastAPI

from app.routers.ai_result_documents import router as ai_result_document_router

app = FastAPI(title="SilverCare AI Backend")
app.include_router(ai_result_document_router)


@app.get("/health")
def health() -> dict[str, str]:
    """서버가 살아 있는지 확인한다."""
    return {"status": "ok"}
