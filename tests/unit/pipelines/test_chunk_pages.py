"""페이지 원문을 조각으로 나누는 순수 함수 테스트. 합성 문서만 쓴다."""

import random
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app.domain.document.document_chunk import DocumentChunk, PageText
from app.pipelines.documents.steps import chunk_pages as chunk_pages_module
from app.pipelines.documents.steps.chunk_pages import (
    MAX_TOKENS,
    chunk_pages,
    count_tokens_cl100k,
    is_personal_info_line,
)

PAGE_ID = UUID("00000000-0000-0000-0000-000000000001")
HEADER = "검사명 결과 단위 참고치"


def count_characters(text: str) -> int:
    """글자 수를 토큰 수로 보는 가짜 계산기. 경계를 손으로 계산하기 쉽다."""
    return len(text)


def make_page(text: str, page_no: int = 1, page_id: UUID = PAGE_ID) -> PageText:
    """합성 원문 페이지 한 장을 만든다."""
    return PageText(page_id=page_id, page_no=page_no, text=text)


def numbered_lines(count: int, width: int = 50) -> list[str]:
    """번호가 붙은 같은 길이의 합성 줄을 만든다."""
    return [f"{index:03d} " + "가" * (width - 4) for index in range(count)]


def chunk(*pages: PageText) -> list[DocumentChunk]:
    """글자 수 계산기로 조각을 나눈다."""
    return chunk_pages(pages, count_tokens=count_characters)


def test_blank_pages_make_no_chunks() -> None:
    """빈 페이지와 공백뿐인 페이지는 조각을 만들지 않는다."""
    assert chunk(make_page(""), make_page("  \n\t\n ", page_no=2)) == []


def test_short_page_becomes_one_chunk() -> None:
    """짧은 페이지는 앞뒤 공백과 빈 줄을 뺀 조각 하나가 된다."""
    chunks = chunk(make_page("  처방전  \n\n복용 안내\n"))

    assert chunks == [
        DocumentChunk(chunk_index=0, page_id=PAGE_ID, content="처방전\n복용 안내", token_count=9)
    ]


def test_long_page_is_split_within_token_range() -> None:
    """긴 페이지는 마지막 조각을 빼고 300~500 토큰 조각으로 나뉜다."""
    chunks = chunk(make_page("\n".join(numbered_lines(40))))

    assert len(chunks) > 1
    assert all(item.token_count <= MAX_TOKENS for item in chunks)
    assert all(item.token_count >= 300 for item in chunks[:-1])
    assert all(item.token_count == len(item.content) for item in chunks)


def test_next_chunk_starts_with_last_two_lines_of_previous() -> None:
    """두 줄이 겹침 상한 안이면 다음 조각은 앞 조각의 마지막 두 줄로 시작한다."""
    chunks = chunk(make_page("\n".join(numbered_lines(60, width=40))))

    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.content.split("\n")[:2] == previous.content.split("\n")[-2:]


def test_overlap_shrinks_to_one_line_over_token_limit() -> None:
    """두 줄이 겹침 상한 100을 넘으면 마지막 한 줄만 겹친다."""
    chunks = chunk(make_page("\n".join(numbered_lines(40, width=60))))

    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.content.split("\n")[0] == previous.content.split("\n")[-1]
        assert current.content.split("\n")[1] != previous.content.split("\n")[-1]


def test_long_line_pieces_do_not_overlap() -> None:
    """긴 줄을 나눈 조각끼리는 겹치지 않아 조각 토큰 합이 원문의 1.2배를 넘지 않는다."""
    long_line = " ".join(["가나다라마바사"] * 200)
    text = "\n".join(["짧은 줄 하나 입니다 " * 20, long_line, long_line])

    chunks = chunk_pages([make_page(text)])

    assert len(chunks) > 1
    assert all(item.token_count <= MAX_TOKENS for item in chunks)
    assert sum(item.token_count for item in chunks) <= 1.2 * count_tokens_cl100k(text)


def test_every_line_is_kept_in_order() -> None:
    """겹친 줄을 빼면 원문의 모든 줄이 순서대로 남는다."""
    lines = numbered_lines(40)
    chunks = chunk(make_page("\n".join(lines)))

    seen: list[str] = []
    for item in chunks:
        seen.extend(line for line in item.content.split("\n") if line not in seen)
    assert seen == lines


def test_table_header_is_repeated_when_table_is_split() -> None:
    """표 안에서 잘리면 다음 조각 앞에 표 머리글 줄이 붙는다."""
    rows = [f"항목{index:02d} 13.{index} g/dL 12.0-16.0" + " 가" * 10 for index in range(30)]
    chunks = chunk(make_page("\n".join([HEADER, *rows])))

    assert len(chunks) > 1
    assert all(item.content.startswith(HEADER + "\n") for item in chunks)


def test_header_is_not_repeated_after_table_ends() -> None:
    """빈 줄로 표가 끝난 뒤의 조각에는 머리글을 붙이지 않는다."""
    rows = [f"항목{index} 13.{index} g/dL" for index in range(3)]
    text = "\n".join([HEADER, *rows, "", *numbered_lines(30)])
    chunks = chunk(make_page(text))

    assert len(chunks) > 1
    assert all(not item.content.startswith(HEADER) for item in chunks[1:])


def test_header_is_not_attached_to_findings_after_table() -> None:
    """빈 줄 없이 표 뒤에 소견 문장이 이어지면 소견 조각에는 머리글을 붙이지 않는다."""
    rows = [f"혈색소 13.{index} g/dL 12.0-16.0" for index in range(3)]
    findings = ["의사 소견 다음 진료 때 다시 확인합니다 추가 설명 문장입니다 " * 3] * 20
    chunks = chunk_pages([make_page("\n".join([HEADER, *rows, *findings]))])

    assert len(chunks) > 1
    assert chunks[0].content.startswith(HEADER)
    assert all(HEADER not in item.content for item in chunks[1:])


def test_header_is_dropped_when_it_would_exceed_limit() -> None:
    """머리글을 붙이면 500을 넘는 조각에는 머리글 없이 행만 담는다(재현 입력 회귀 테스트)."""
    header = "검사명 결과" + " A" * 245
    row = "X 1" + " B" * 247

    chunks = chunk_pages([make_page(f"{header}\n{row}")])

    assert [item.token_count <= MAX_TOKENS for item in chunks] == [True, True]
    assert chunks[1].content == row


@pytest.mark.parametrize(
    "line",
    [
        "검사명\t결과\t단위\t참고치",
        "검사명    결과     단위   참고치",
        "검사명|결과|단위|참고치",
        "| 검사명 | 결과 | 단위 | 참고치 |",
        "검사 명 결과 단위 참고치",
        "Test Result Unit Reference",
        "TEST   RESULT   UNITS   REFERENCE RANGE",
        "No 약품명 용법 용량 투약일수 비고",
    ],
)
def test_ocr_header_variants_are_headers(line: str) -> None:
    """탭·여러 칸 공백·| 구분, 일부 칸 띄어쓰기가 깨진 머리글, 영어 머리글을 머리글로 본다."""
    assert chunk_pages_module._is_table_header(line) is True


@pytest.mark.parametrize(
    "line",
    [
        "검사 결과 단위를 확인해 주세요",
        "항목 결과 모두 정상이며 추가 검사는 필요하지 않습니다",
        "결과 판정은 다음 진료 때 설명드립니다.",
        "Please check the test result with your doctor.",
        "혈색소 13.2 g/dL 12.0-16.0",
    ],
)
def test_sentences_with_header_words_are_not_headers(line: str) -> None:
    """칸 이름이 우연히 든 일반 문장과 표 행은 머리글로 보지 않는다."""
    assert chunk_pages_module._is_table_header(line) is False


@pytest.mark.parametrize("line", ["검사명결과단위참고치", "검 사 명  결과값  단 위"])
def test_badly_spaced_header_is_not_detected(line: str) -> None:
    """한계: 칸 사이 띄어쓰기가 사라지거나 칸 이름이 두 개 이상 쪼개진 머리글은 알아보지 못한다.

    이때는 머리글이 반복되지 않을 뿐 줄 자체는 조각에 그대로 들어간다.
    """
    assert chunk_pages_module._is_table_header(line) is False


@pytest.mark.parametrize("row", ["혈색소|13.2|g/dL|12.0-16.0", "혈색소\t13.2\tg/dL\t12.0-16.0"])
def test_pipe_and_tab_rows_are_table_rows(row: str) -> None:
    """| 나 탭으로 칸을 나눈 행도 표 행으로 본다."""
    assert chunk_pages_module._is_table_row(row) is True


def test_pipe_table_header_is_repeated_when_split() -> None:
    """| 로 칸을 나눈 표도 잘리면 다음 조각 앞에 머리글이 붙는다."""
    header = "검사명|결과|단위|참고치"
    rows = [f"항목{index:02d}|13.{index}|g/dL|12.0-16.0" + "|가" * 10 for index in range(30)]

    chunks = chunk(make_page("\n".join([header, *rows])))

    assert len(chunks) > 1
    assert all(item.content.startswith(header + "\n") for item in chunks)


def near(rng: random.Random, edge: int) -> int:
    """절반은 1~edge+20 사이 아무 값, 절반은 edge 바로 근처 값을 고른다."""
    return rng.randint(1, edge + 20) if rng.random() < 0.5 else edge + rng.randint(-2, 2)


def varied_page_text(rng: random.Random, header_edge: int, row_edge: int) -> str:
    """긴 머리글, 긴 행, 공백 없는 긴 줄, 소견, 빈 줄을 섞은 합성 원문 한 쪽을 만든다.

    edge는 머리글·행이 줄 하나 상한(250)에 딱 닿는 반복 수다. 경계를 넘는 행은 나뉘고,
    넘는 머리글은 일반 줄이 된다.
    """
    makers = [
        lambda: "검사명 결과" + " A" * near(rng, header_edge),
        lambda: "X 10" + " B" * near(rng, row_edge),
        lambda: f"혈색소 13.{rng.randint(0, 9)} g/dL 12.0-16.0",
        lambda: "혈" * rng.randint(300, 1200),
        lambda: "소견 다음 진료 때 다시 확인합니다 " * rng.randint(1, 15),
        lambda: "",
    ]
    weights = [2, 4, 4, 1, 2, 1]
    return "\n".join(rng.choices(makers, weights)[0]() for _ in range(rng.randint(1, 40)))


def varied_pages(seed: int, *, header_edge: int, row_edge: int) -> list[PageText]:
    """시드마다 1~3쪽짜리 합성 문서를 만든다. 같은 시드는 같은 문서다."""
    rng = random.Random(seed)
    return [
        make_page(varied_page_text(rng, header_edge, row_edge), page_no=no)
        for no in range(1, rng.randint(1, 3) + 1)
    ]


@pytest.mark.parametrize("seed", range(200))
def test_all_chunks_stay_within_limit_on_varied_inputs(seed: int) -> None:
    """여러 입력 조합에서도 모든 조각이 500 이하이고 token_count가 내용과 맞는다."""
    # 글자 수 기준으로 머리글은 6+2*122=250자, 행은 4+2*123=250자에서 경계에 닿는다.
    pages = varied_pages(seed, header_edge=122, row_edge=123)

    chunks = chunk_pages(pages, count_tokens=count_characters)

    assert all(item.token_count <= MAX_TOKENS for item in chunks)
    assert all(item.token_count == len(item.content) for item in chunks)


@pytest.mark.parametrize("seed", range(50))
def test_all_chunks_stay_within_limit_with_real_tokenizer(seed: int) -> None:
    """실제 cl100k_base 토큰 수로도 여러 입력 조합에서 모든 조각이 500 이하다."""
    # cl100k 기준으로 머리글은 " A" 245개, 행은 " B" 247개 근처에서 250토큰에 닿는다.
    chunks = chunk_pages(varied_pages(seed, header_edge=245, row_edge=247))

    assert all(item.token_count <= MAX_TOKENS for item in chunks)
    assert all(item.token_count == count_tokens_cl100k(item.content) for item in chunks)


def test_chunk_index_is_continuous_across_pages() -> None:
    """조각 번호는 페이지를 넘어 0부터 이어지고, 조각마다 자기 페이지를 가리킨다."""
    first_id, second_id = uuid4(), uuid4()
    chunks = chunk(
        make_page("\n".join(numbered_lines(20)), page_no=1, page_id=first_id),
        make_page("두 번째 페이지", page_no=2, page_id=second_id),
    )

    assert [item.chunk_index for item in chunks] == list(range(len(chunks)))
    assert chunks[-1].page_id == second_id
    assert {item.page_id for item in chunks[:-1]} == {first_id}


def test_long_line_without_spaces_is_split_on_characters() -> None:
    """공백 없는 긴 줄도 글자 단위로 나뉘어 상한을 넘지 않는다."""
    chunks = chunk(make_page("혈" * 2000))

    assert len(chunks) > 1
    assert all(item.token_count <= MAX_TOKENS for item in chunks)
    assert set("".join(item.content.replace("\n", "") for item in chunks)) == {"혈"}


def test_excluded_lines_are_left_out() -> None:
    """exclude_line이 고른 줄은 조각에 넣지 않는다."""
    chunks = chunk_pages(
        [make_page("성명 홍길동\n혈색소 13.2")],
        count_tokens=count_characters,
        exclude_line=lambda line: line.startswith("성명"),
    )

    assert [item.content for item in chunks] == ["혈색소 13.2"]


# 합성 값만 쓴다. 실제 사람·번호가 아니다.
@pytest.mark.parametrize(
    "line",
    [
        "성명: 홍길동",
        "성 명 홍길동",
        "환자명 홍길동",
        "이름 : 홍길동",
        "생년월일 1940-01-01",
        "주민등록번호 400101-1******",
        "주민번호: 400101-1000000",
        "등록번호 12345678",
        "환자번호: A0001",
        "연락처 010-0000-0000",
        "전화번호 02-000-0000",
        "휴대폰: 01000000000",
        "주소: 서울시 어딘가 1",
        "[성명] 홍길동",
        "• 주소 서울시 어딘가",
        "성명: 홍길동   성별: 남   나이: 78",
        "보호자 400101-2000000",
        "문의 010 0000 0000",
        "병원 대표번호 031-000-0000",
    ],
)
def test_personal_info_lines_are_detected(line: str) -> None:
    """개인정보 라벨로 시작하거나 주민등록번호·전화번호 형식이 든 줄을 고른다."""
    assert is_personal_info_line(line) is True


@pytest.mark.parametrize(
    "line",
    [
        "혈색소 13.2 g/dL 12.0-16.0",
        "검사명 결과 단위 참고치",
        "처방: 아스피린 100mg 1일 1회 30일",
        "성별: 남   나이: 78",
        "성별 여 / 나이 81세",
        "검사일 2024-01-15",
        "Na 140 mmol/L 135-145",
        "전화로 문의하세요",
        "주소견: 특이 소견 없음",
        "이름표를 지참하세요",
        "백혈구 6.5 10^3/uL 4.0-10.0",
    ],
)
def test_clinical_lines_are_kept(line: str) -> None:
    """검사값·처방 줄과 성별·나이만 있는 줄, 라벨이 다른 낱말의 일부인 줄은 남긴다."""
    assert is_personal_info_line(line) is False


def test_personal_info_filter_keeps_results_in_chunks() -> None:
    """필터를 넘기면 개인정보 줄은 빠지고 검사 결과와 성별·나이 줄은 조각에 남는다."""
    text = "\n".join(
        [
            "성명: 홍길동",
            "성별: 남   나이: 78",
            "연락처 010-0000-0000",
            HEADER,
            "혈색소 13.2 g/dL 12.0-16.0",
        ]
    )

    chunks = chunk_pages(
        [make_page(text)], count_tokens=count_characters, exclude_line=is_personal_info_line
    )

    assert [item.content for item in chunks] == [
        "\n".join(["성별: 남   나이: 78", HEADER, "혈색소 13.2 g/dL 12.0-16.0"])
    ]


def test_same_input_gives_same_chunks() -> None:
    """같은 입력은 항상 같은 조각을 만든다."""
    page = make_page("\n".join(numbered_lines(40)))

    assert chunk(page) == chunk(page)


def test_real_tokenizer_keeps_chunks_within_limit() -> None:
    """실제 cl100k_base 토큰 수로도 조각이 상한을 넘지 않고 token_count가 맞는다."""
    rows = [f"항목{index:02d} 혈색소 13.{index % 10} g/dL 12.0-16.0 정상" for index in range(120)]
    chunks = chunk_pages([make_page("\n".join([HEADER, *rows]))])

    assert len(chunks) > 1
    assert all(item.token_count <= MAX_TOKENS for item in chunks)
    assert all(item.token_count == count_tokens_cl100k(item.content) for item in chunks)


def test_special_token_text_is_counted_as_plain_text() -> None:
    """특수 토큰처럼 생긴 글자도 예외 없이 일반 글자로 센다."""
    assert count_tokens_cl100k("<|endoftext|>") > 1


def test_chunker_does_not_import_db_settings_or_api() -> None:
    """청커는 DB, 설정, 외부 API 모듈을 import하지 않는다."""
    source = Path(chunk_pages_module.__file__).read_text(encoding="utf-8")

    for forbidden in ("sqlalchemy", "app.core", "app.infrastructure", "openai", "httpx"):
        assert forbidden not in source
