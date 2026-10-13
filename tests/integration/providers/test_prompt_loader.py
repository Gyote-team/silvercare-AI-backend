"""프롬프트 읽기 테스트. 저장소의 실제 prompts/ 파일을 읽는다."""

import pytest

from app.infrastructure.providers.llm.prompt_loader import load_prompt


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_loads_document_extraction_prompt(version: str) -> None:
    """문서 추출 프롬프트를 버전별로 읽는다. 예전 버전 파일도 남아 있다."""
    prompt = load_prompt("document_extraction", version)

    assert prompt.startswith(f"# 의료 문서 페이지 추출 {version}")


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_prompt_forbids_medical_judgment(version: str) -> None:
    """의료 판단 금지, 문서에 적힌 것만 옮기라는 지시가 들어 있다."""
    prompt = load_prompt("document_extraction", version)

    assert "의료 판단을 하지 않는다" in prompt
    assert "문서에 적힌 것만 옮긴다" in prompt


def test_v2_prompt_asks_for_raw_text_with_masked_resident_number() -> None:
    """v2에는 페이지 원문을 옮기고 주민등록번호 뒷자리를 가리라는 지시가 들어 있다."""
    prompt = load_prompt("document_extraction", "v2")

    assert "## raw_text" in prompt
    assert "뒷자리 7자리를 *로 가린다" in prompt


def test_missing_version_raises() -> None:
    """없는 버전은 FileNotFoundError다."""
    with pytest.raises(FileNotFoundError):
        load_prompt("document_extraction", "v999")
