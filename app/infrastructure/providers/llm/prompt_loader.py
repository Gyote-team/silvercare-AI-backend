"""prompts/ 폴더에서 기능·버전별 프롬프트 파일을 읽는다."""

from pathlib import Path

# 저장소 루트의 prompts/ 폴더. 이 파일은 app/infrastructure/providers/llm/ 아래에 있다.
_PROMPTS_ROOT = Path(__file__).resolve().parents[4] / "prompts"


def load_prompt(feature: str, version: str) -> str:
    """prompts/{feature}/{version}.md 내용을 반환한다. 파일이 없으면 FileNotFoundError다."""
    return (_PROMPTS_ROOT / feature / f"{version}.md").read_text(encoding="utf-8")
