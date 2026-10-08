"""FastAPI 앱을 만들고 상태 확인 API를 등록한다."""

from fastapi import FastAPI

app = FastAPI(title="SilverCare AI Backend")


@app.get("/health")
def health() -> dict[str, str]:
    """서버가 살아 있는지 확인한다."""
    return {"status": "ok"}
