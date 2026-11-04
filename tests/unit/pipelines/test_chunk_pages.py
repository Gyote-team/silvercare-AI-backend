"""페이지 원문을 조각으로 나누는 순수 함수 테스트. 합성 문서만 쓴다."""

from pathlib import Path
from uuid import UUID, uuid4

from app.domain.document.document_chunk import DocumentChunk, PageText
from app.pipelines.documents.steps import chunk_pages as chunk_pages_module
from app.pipelines.documents.steps.chunk_pages import (
    MAX_TOKENS,
    chunk_pages,
    count_tokens_cl100k,
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
