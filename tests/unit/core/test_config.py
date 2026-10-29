"""서버 설정의 기본값을 검증한다."""

from app.core.config import Settings


def _default(name: str) -> object:
    """.env와 환경 변수를 읽지 않고 필드 기본값만 꺼낸다."""
    return Settings.model_fields[name].default


def test_stale_minutes_default_exceeds_worst_analysis_time() -> None:
    """멈춘 분석 판단 시간의 기본값이 기본 설정의 최악 분석 시간보다 길다."""
    attempts = _default("openai_vlm_max_retries") + 1
    per_attempt = _default("openai_vlm_connect_timeout_seconds") + _default(
        "openai_vlm_timeout_seconds"
    )
    backoff = _default("openai_vlm_retry_backoff_seconds") * (
        2 ** _default("openai_vlm_max_retries") - 1
    )
    worst_seconds = _default("document_max_pages") * (attempts * per_attempt + backoff)

    assert _default("document_analysis_stale_minutes") * 60 > worst_seconds
