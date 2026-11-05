"""페이지 원문을 검색용 조각으로 나눈다. DB나 외부 API를 쓰지 않는 순수 함수다."""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache

import tiktoken

from app.domain.document.document_chunk import DocumentChunk, PageText

MAX_TOKENS = 500
# 다음 조각 앞에 다시 넣는 앞 조각의 마지막 줄 수와, 그 줄들의 토큰 상한.
OVERLAP_LINES = 2
OVERLAP_MAX_TOKENS = 100
# 한 줄이 이보다 길면 공백, 그다음 글자 단위로 나눈다. 머리글과 겹친 줄이 들어갈 자리를 남긴다.
_MAX_LINE_TOKENS = MAX_TOKENS // 2
# 표 머리글 칸 이름. 원문은 표를 칸 사이 공백·탭·|로 옮기므로 칸 이름으로 머리글 줄을 알아본다.
# 영어 칸 이름은 소문자로 적고 비교할 때 소문자로 바꾼다.
_HEADER_CELLS = frozenset(
    {
        "test",
        "result",
        "unit",
        "units",
        "reference",
        "range",
        "value",
        "flag",
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
# 표 줄로 보는 최소 칸 수. 표 줄은 숫자도 하나 이상 있어야 한다.
_MIN_ROW_CELLS = 3
_CELL_SEPARATOR = re.compile(r"[\s|]+")
# 문장 끝맺음. 칸 이름이 우연히 두 개 든 안내 문장을 머리글로 보지 않는 데 쓴다.
_SENTENCE_END = re.compile(r"(?:[다요]\s*[.!?]?|[.!?])$")


# 라벨 정규식 공통 규칙: 라벨 글자 사이 OCR 공백을 허용하고, 라벨 바로 뒤에 한글이 오면
# ("전화로", "주소견") 다른 낱말로 보고 고르지 않는다. 앞의 기호(•, [ 등)는 건너뛴다.
# 이 라벨로 시작하는 줄은 내용과 상관없이 거른다.
_PERSONAL_LABEL = re.compile(
    r"^[^\w]*(?:"
    r"성\s*명|환\s*자\s*명|이\s*름|생\s*년\s*월\s*일"
    r"|주\s*민\s*(?:등\s*록\s*)?번\s*호|등\s*록\s*번\s*호|환\s*자\s*번\s*호"
    r"|주\s*소\s*지|거\s*주\s*지"
    r")(?![가-힣])"
)
# 연락처 라벨. "전화 상담 예정" 같은 안내 문장을 남기려고 줄에 숫자가 있을 때만 거른다.
_CONTACT_LABEL = re.compile(
    r"^[^\w]*(?:연\s*락\s*처|전\s*화(?:\s*번\s*호)?|휴\s*대\s*(?:폰|전\s*화))(?![가-힣])"
)
# "주소"는 진료기록에서 주호소(chief complaint)로도 쓰여, 줄에 주소 형태가 있을 때만 거른다.
_ADDRESS_LABEL = re.compile(r"^[^\w]*주\s*소(?![가-힣])")
# 주소 형태: 광역 시·도 이름으로 시작해 시·군·구가 이어지거나, 도로명 뒤에 건물 번호가 온다.
# 도로명은 조사 "으로"를 빼고, 번호 바로 뒤에 한글(일, 회, 주 등)이 오면 기간·횟수로 보고
# 주소로 보지 않는다("고열로 2일 전부터").
_ADDRESS_FORM = re.compile(
    r"(?:서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|충청|전북|전남|전라"
    r"|경북|경남|경상|제주)[가-힣]*\s*[가-힣]+(?:시|군|구)"
    r"|[가-힣]+(?<!으)(?:로|길)\s?\d+(?:-\d+)?(?![\d가-힣])"
)
# 주민등록번호(뒷자리 마스킹 포함), 휴대전화, 하이픈 있는 유선 번호. 앞뒤가 숫자면 고르지 않는다.
_PERSONAL_NUMBER = re.compile(
    r"(?<!\d)(?:"
    r"\d{6}\s*-\s*[1-8][\d*]{6}"
    r"|01[016789][-\s]?\d{3,4}[-\s]?\d{4}"
    r"|0\d{1,2}-\d{3,4}-\d{4}"
    r")(?!\d)"
)


def is_personal_info_line(line: str) -> bool:
    """개인정보 줄이면 True다.

    성명·생년월일 같은 라벨로 시작하는 줄, 숫자가 있는 연락처 라벨 줄, 주소 형태가 있는
    "주소" 라벨 줄, 주민등록번호·전화번호 형식이 든 줄을 고른다.
    검사값·처방 줄, 성별·나이만 있는 줄, 주호소("주소: 흉통")와 "전화 상담" 안내는 남긴다.
    줄 중간에 나오는 라벨은 보지 않는다.
    """
    return bool(
        _PERSONAL_LABEL.match(line)
        or (_CONTACT_LABEL.match(line) and any(char.isdigit() for char in line))
        or (_ADDRESS_LABEL.match(line) and _ADDRESS_FORM.search(line))
        or _PERSONAL_NUMBER.search(line)
    )


@lru_cache
def _encoding() -> tiktoken.Encoding:
    # text-embedding-3-small과 같은 인코딩이다. 처음 부를 때 인코딩 파일을 읽어 둔다.
    return tiktoken.get_encoding("cl100k_base")


def count_tokens_cl100k(text: str) -> int:
    """임베딩 모델 기준 토큰 수를 반환한다. 특수 토큰처럼 생긴 글자도 일반 글자로 센다."""
    return len(_encoding().encode_ordinary(text))


@dataclass(frozen=True)
class _Line:
    """조각을 만드는 단위 줄. 긴 줄은 여러 _Line으로 나뉜다."""

    text: str
    # 이 줄이 속한 표의 머리글 줄. 표 밖이거나 이 줄이 머리글이면 None이다.
    header: str | None
    # 긴 한 줄을 나눈 조각 중 두 번째 이후면 True다. 이 앞에서는 겹치지 않는다.
    continued: bool = False


def chunk_pages(
    pages: Sequence[PageText],
    *,
    count_tokens: Callable[[str], int] = count_tokens_cl100k,
    exclude_line: Callable[[str], bool] | None = None,
) -> list[DocumentChunk]:
    """페이지 순서대로 원문을 줄 경계에서 나눈 조각 목록을 반환한다.

    조각은 MAX_TOKENS를 넘지 않게 줄을 쌓고, 다음 조각은 앞 조각의 마지막 줄 1~2개
    (OVERLAP_MAX_TOKENS 이하)로 시작한다. 긴 한 줄을 나눈 조각끼리는 겹치지 않는다.
    표 줄에서 잘리면 다음 조각 앞에 표 머리글 줄을 붙이고, 붙이면 상한을 넘는 조각에는
    붙이지 않는다. 페이지는 합치지 않는다.
    exclude_line이 True를 돌려준 줄은 조각에서 뺀다(색인에서는 is_personal_info_line을 넘긴다).
    모든 조각은 MAX_TOKENS 이하이고, 어긋나면 잘라 원문을 잃는 대신 ValueError를 던진다.
    """
    chunks: list[DocumentChunk] = []
    for page in pages:
        lines = _read_lines(page.text, count_tokens, exclude_line)
        for window in _windows(lines, count_tokens):
            content = _render(window)
            token_count = count_tokens(content)
            if token_count > MAX_TOKENS:
                raise ValueError("조각이 토큰 상한을 넘었습니다")
            chunks.append(
                DocumentChunk(
                    chunk_index=len(chunks),
                    page_id=page.page_id,
                    content=content,
                    token_count=token_count,
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
        # 빈 줄 없이 소견 같은 문장이 이어지는 경우가 많아, 표 줄이 아니면 표가 끝난 것으로 본다.
        if not _is_table_row(line):
            header = None
        pieces = _split_long_line(line, count_tokens)
        lines.extend(
            _Line(piece, header, continued=index > 0) for index, piece in enumerate(pieces)
        )
    return lines


def _cells(line: str) -> list[str]:
    return [cell for cell in _CELL_SEPARATOR.split(line) if cell]


def _is_table_header(line: str) -> bool:
    # 끝맺음으로 문장을 거르는 단순 규칙이라, 끝맺음 없이 칸 이름 두 개가 든 문장은 못 거른다.
    if _SENTENCE_END.search(line):
        return False
    return sum(cell.lower() in _HEADER_CELLS for cell in _cells(line)) >= _MIN_HEADER_CELLS


def _is_table_row(line: str) -> bool:
    return len(_cells(line)) >= _MIN_ROW_CELLS and any(char.isdigit() for char in line)


def _windows(lines: list[_Line], count_tokens: Callable[[str], int]) -> list[list[_Line]]:
    windows: list[list[_Line]] = []
    current: list[_Line] = []
    for line in lines:
        if not current:
            current = _start(line, count_tokens)
        elif count_tokens(_render([*current, line])) > MAX_TOKENS:
            windows.append(current)
            tail = _overlap(current, line, count_tokens)
            current = [*tail, line] if tail else _start(line, count_tokens)
        else:
            current.append(line)
    if current:
        windows.append(current)
    return windows


def _start(line: _Line, count_tokens: Callable[[str], int]) -> list[_Line]:
    # 머리글을 붙이면 상한을 넘을 때는 머리글을 빼고 시작한다. 머리글 줄은 같은 페이지 앞 조각에
    # 이미 있어 내용이 빠지지 않는다. 머리글 없는 줄 하나는 _MAX_LINE_TOKENS 이하다.
    if line.header is not None and count_tokens(_render([line])) > MAX_TOKENS:
        return [replace(line, header=None)]
    return [line]


def _overlap(previous: list[_Line], line: _Line, count_tokens: Callable[[str], int]) -> list[_Line]:
    # 긴 줄을 나눈 조각끼리 겹치면 같은 글이 거의 두 번씩 임베딩되므로 겹치지 않는다.
    if line.continued:
        return []
    # 앞 조각 전체를 되풀이하지 않게 첫 줄은 남기지 않는다. 넘치면 겹치는 줄을 줄인다.
    for size in range(min(OVERLAP_LINES, len(previous) - 1), 0, -1):
        tail = previous[-size:]
        if (
            count_tokens("\n".join(item.text for item in tail)) <= OVERLAP_MAX_TOKENS
            and count_tokens(_render([*tail, line])) <= MAX_TOKENS
        ):
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
