"""상태 확인 API 통합 테스트."""

from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_ok() -> None:
    """/health가 200과 status ok를 반환한다."""
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
