"""프롬프트 읽기 테스트. 저장소의 실제 prompts/ 파일을 읽는다."""

import pytest

from app.infrastructure.providers.llm.prompt_loader import load_prompt


def test_loads_document_extraction_v1() -> None:
    """문서 추출 v1 프롬프트를 읽는다."""
    prompt = load_prompt("document_extraction", "v1")

    assert prompt.startswith("# 의료 문서 페이지 추출 v1")


def test_prompt_forbids_medical_judgment() -> None:
    """의료 판단 금지, 문서에 적힌 것만 옮기라는 지시가 들어 있다."""
    prompt = load_prompt("document_extraction", "v1")

    assert "의료 판단을 하지 않는다" in prompt
    assert "문서에 적힌 것만 옮긴다" in prompt


def test_missing_version_raises() -> None:
    """없는 버전은 FileNotFoundError다."""
    with pytest.raises(FileNotFoundError):
        load_prompt("document_extraction", "v999")
