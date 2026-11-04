"""페이지 원문을 검색용 조각으로 나눈다. DB나 외부 API를 쓰지 않는 순수 함수다."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from app.domain.document.document_chunk import DocumentChunk, PageText

MAX_TOKENS = 500
# 다음 조각 앞에 다시 넣는 앞 조각의 마지막 줄 수.
OVERLAP_LINES = 2
# 한 줄이 이보다 길면 공백, 그다음 글자 단위로 나눈다. 머리글과 겹친 줄이 들어갈 자리를 남긴다.
_MAX_LINE_TOKENS = MAX_TOKENS // 2
# 표 머리글 칸 이름. 원문은 표를 칸 사이 공백으로만 옮기므로 칸 이름으로 머리글 줄을 알아본다.
_HEADER_CELLS = frozenset(
    {
        "검사명",
        "검사항목",
        "항목",
        "결과",
        "결과값",
        "단위",
        "참고치",
        "참고범위",
        "정상범위",
        "판정",
        "약품명",
        "약명",
        "용법",
        "용량",
        "투약일수",
    }
)
_MIN_HEADER_CELLS = 2


@lru_cache
def _encoding() -> tiktoken.Encoding:
    # text-embedding-3-small과 같은 인코딩이다. 처음 부를 때 인코딩 파일을 읽어 둔다.
    return tiktoken.get_encoding("cl100k_base")


def count_tokens_cl100k(text: str) -> int:
    """임베딩 모델 기준 토큰 수를 반환한다. 특수 토큰처럼 생긴 글자도 일반 글자로 센다."""
    return len(_encoding().encode_ordinary(text))


@dataclass(frozen=True)
class _Line:
    text: str
    # 이 줄이 속한 표의 머리글 줄. 표 밖이거나 이 줄이 머리글이면 None이다.
    header: str | None


def chunk_pages(
    pages: Sequence[PageText],
    *,
    count_tokens: Callable[[str], int] = count_tokens_cl100k,
    exclude_line: Callable[[str], bool] | None = None,
) -> list[DocumentChunk]:
    """페이지 순서대로 원문을 줄 경계에서 나눈 조각 목록을 반환한다.

    조각은 MAX_TOKENS를 넘지 않게 줄을 쌓고, 다음 조각은 앞 조각의 마지막 줄 1~2개로 시작한다.
    표 안에서 잘리면 다음 조각 앞에 표 머리글 줄을 붙인다. 페이지는 합치지 않는다.
    exclude_line은 조각에서 뺄 줄(예: 개인정보 머리글)을 고르는 자리로, 지금은 쓰지 않는다.
    """
    chunks: list[DocumentChunk] = []
    for page in pages:
        lines = _read_lines(page.text, count_tokens, exclude_line)
        for window in _windows(lines, count_tokens):
            content = _render(window)
            chunks.append(
                DocumentChunk(
                    chunk_index=len(chunks),
                    page_id=page.page_id,
                    content=content,
                    token_count=count_tokens(content),
                )
            )
    return chunks


def _read_lines(
    text: str,
    count_tokens: Callable[[str], int],
    exclude_line: Callable[[str], bool] | None,
) -> list[_Line]:
    lines: list[_Line] = []
    header: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            # 빈 줄에서 표가 끝난 것으로 본다.
            header = None
            continue
        if exclude_line is not None and exclude_line(line):
            continue
        # 긴 문장이 칸 이름을 우연히 담은 경우는 머리글로 보지 않고 나눈다.
        if _is_table_header(line) and count_tokens(line) <= _MAX_LINE_TOKENS:
            header = line
            lines.append(_Line(line, None))
            continue
        lines.extend(_Line(piece, header) for piece in _split_long_line(line, count_tokens))
    return lines


def _is_table_header(line: str) -> bool:
    return sum(cell in _HEADER_CELLS for cell in line.split()) >= _MIN_HEADER_CELLS


def _windows(lines: list[_Line], count_tokens: Callable[[str], int]) -> list[list[_Line]]:
    windows: list[list[_Line]] = []
    current: list[_Line] = []
    for line in lines:
        if current and count_tokens(_render([*current, line])) > MAX_TOKENS:
            windows.append(current)
            current = [*_overlap(current, line, count_tokens), line]
        else:
            current.append(line)
    if current:
        windows.append(current)
    return windows


def _overlap(
    previous: list[_Line], line: _Line, count_tokens: Callable[[str], int]
) -> list[_Line]:
    # 앞 조각 전체를 되풀이하지 않게 첫 줄은 남기지 않는다. 넘치면 겹치는 줄을 줄인다.
    for size in range(min(OVERLAP_LINES, len(previous) - 1), 0, -1):
        tail = previous[-size:]
        if count_tokens(_render([*tail, line])) <= MAX_TOKENS:
            return tail
    return []


def _render(window: list[_Line]) -> str:
    lines = [line.text for line in window]
    # 첫 줄이 표 안의 줄이면 그 표의 머리글은 이 조각에 없으므로 앞에 붙인다.
    if window[0].header is not None:
        lines.insert(0, window[0].header)
    return "\n".join(lines)


def _split_long_line(line: str, count_tokens: Callable[[str], int]) -> list[str]:
    if count_tokens(line) <= _MAX_LINE_TOKENS:
        return [line]
    pieces: list[str] = []
    current = ""
    for word in line.split():
        candidate = f"{current} {word}" if current else word
        if count_tokens(candidate) <= _MAX_LINE_TOKENS:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if count_tokens(word) <= _MAX_LINE_TOKENS:
            current = word
        else:
            *head, current = _split_characters(word, count_tokens)
            pieces.extend(head)
    if current:
        pieces.append(current)
    return pieces


def _split_characters(word: str, count_tokens: Callable[[str], int]) -> list[str]:
    # 토큰 id가 아니라 글자 단위로 자르므로 한글 글자가 깨지지 않는다.
    pieces: list[str] = []
    current = ""
    for character in word:
        if current and count_tokens(current + character) > _MAX_LINE_TOKENS:
            pieces.append(current)
            current = character
        else:
            current += character
    pieces.append(current)
    return pieces
