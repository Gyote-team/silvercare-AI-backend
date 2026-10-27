"""서버 설정의 기본값을 검증한다."""

from app.core.config import Settings


def test_stale_minutes_defaults_to_60() -> None:
    """멈춘 분석 판단 시간의 기본값은 60분이다. .env와 환경 변수는 읽지 않는다."""
    assert Settings.model_fields["document_analysis_stale_minutes"].default == 60
