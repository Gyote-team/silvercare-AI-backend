"""검증된 구조화 답변을 제한된 Markdown으로 변환한다."""

import re

from app.contracts.ai_result_document import AnswerSection


def _safe(text: str) -> str:
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"!?\[[^\]]*\]\([^)]*\)", "", text)
    return text.strip()


def render_markdown(title: str, summary: str, sections: list[AnswerSection]) -> str:
    lines = [f"# {_safe(title)}", "", _safe(summary)]
    for section in sections:
        lines.extend(["", f"## {_safe(section.heading)}", ""])
        for item in section.items:
            text = _safe(item.text)
            if item.kind == "BULLET":
                lines.append(f"- {text}")
            elif item.kind == "CALLOUT":
                lines.append(f"> {text}")
            else:
                lines.extend([text, ""])
    return "\n".join(lines).strip()
