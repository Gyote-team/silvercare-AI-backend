"""페이지 추출 결과 형식 단위 테스트. 값은 모두 합성 데이터다."""

from typing import Any

import pytest
from pydantic import ValidationError

from app.domain.document.page_extraction import (
    DocumentType,
    ExtractedItem,
    ExtractedItemType,
    NormalizedBox,
    PageExtraction,
    SourceBox,
    mask_resident_numbers,
)

RAW_TEXT = "혈색소 13.5 g/dL\n다음 진료 2026-11-03"


def make_item(**overrides: Any) -> dict[str, Any]:
    """합성 검사값 항목 하나를 dict로 만든다. 넘긴 값으로 필드를 바꾼다."""
    item: dict[str, Any] = {
        "item_type": "LAB_VALUE",
        "normalized_name": "혈색소",
        "raw_value": "13.5",
        "normalized_value": "13.5",
        "unit": "g/dL",
        "reference_range": "12.0-16.0",
        "confidence": 0.9,
        "source_text": "혈색소 13.5 g/dL",
        "source_box": {"x": 10, "y": 20, "width": 300, "height": 40},
    }
    return {**item, **overrides}


def make_page(**overrides: Any) -> dict[str, Any]:
    """합성 항목 하나가 든 페이지 결과를 dict로 만든다. 넘긴 값으로 필드를 바꾼다."""
    page: dict[str, Any] = {
        "document_type": "LAB_RESULT",
        "raw_text": RAW_TEXT,
        "items": [make_item()],
    }
    return {**page, **overrides}


def test_parses_valid_page() -> None:
    """올바른 결과는 Enum과 중첩 모델로 읽히고 원문을 그대로 담는다."""
    page = PageExtraction.model_validate(make_page())

    assert page.document_type is DocumentType.LAB_RESULT
    assert page.raw_text == RAW_TEXT
    assert page.has_text is True
    assert page.items[0].item_type is ExtractedItemType.LAB_VALUE
    assert page.items[0].source_box == SourceBox(x=10, y=20, width=300, height=40)


def test_empty_items_is_allowed() -> None:
    """옮길 항목이 없는 페이지는 빈 목록으로 받는다."""
    page = PageExtraction.model_validate(make_page(document_type="UNKNOWN", items=[]))

    assert page.items == []


def test_optional_values_accept_null() -> None:
    """source_text와 item_type 말고는 모두 null이어도 된다."""
    item = ExtractedItem.model_validate(
        make_item(
            normalized_name=None,
            raw_value=None,
            normalized_value=None,
            unit=None,
            reference_range=None,
            confidence=None,
            source_box=None,
        )
    )

    assert item.source_box is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_text": ""},
        {"source_text": "   "},
        {"item_type": "SURGERY"},
        {"confidence": 1.1},
        {"confidence": -0.1},
        {"normalized_name": "가" * 201},
        {"raw_value": "가" * 501},
        {"normalized_value": "가" * 501},
        {"unit": "가" * 51},
        {"reference_range": "가" * 101},
        {"medical_advice": "모르는 필드"},
    ],
)
def test_invalid_item_is_rejected(overrides: dict[str, Any]) -> None:
    """빈 근거, 모르는 종류, 범위·길이 초과, 모르는 필드는 거절한다."""
    with pytest.raises(ValidationError):
        ExtractedItem.model_validate(make_item(**overrides))


@pytest.mark.parametrize("missing_key", ["source_text", "item_type", "unit", "source_box"])
def test_missing_key_is_rejected(missing_key: str) -> None:
    """null을 허용하는 필드도 키 자체가 없으면 거절한다."""
    item = make_item()
    del item[missing_key]

    with pytest.raises(ValidationError):
        ExtractedItem.model_validate(item)


def test_unknown_document_type_is_rejected() -> None:
    """documents.document_type에 없는 문서 종류는 거절한다."""
    with pytest.raises(ValidationError):
        PageExtraction.model_validate(make_page(document_type="RECEIPT", items=[]))


def test_missing_raw_text_is_rejected() -> None:
    """raw_text 키가 없으면 거절한다."""
    page = make_page()
    del page["raw_text"]

    with pytest.raises(ValidationError):
        PageExtraction.model_validate(page)


@pytest.mark.parametrize("raw_text", ["", "   ", "\n"])
def test_blank_raw_text_with_items_is_rejected(raw_text: str) -> None:
    """원문이 비었는데 항목이 있으면 앞뒤가 맞지 않으므로 거절한다."""
    with pytest.raises(ValidationError):
        PageExtraction.model_validate(make_page(raw_text=raw_text))


@pytest.mark.parametrize("raw_text", ["", "   ", "\n"])
def test_blank_page_without_items_is_allowed(raw_text: str) -> None:
    """원문도 항목도 없는 빈 페이지는 받고, 글자가 없다고 알려 준다."""
    page = PageExtraction.model_validate(make_page(raw_text=raw_text, items=[]))

    assert page.has_text is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("주민등록번호 500101-1234567", "주민등록번호 500101-*******"),
        ("500101 - 1234567", "500101 - *******"),
        ("500101–1234567", "500101–*******"),
        ("5001011234567", "500101*******"),
        (
            "환자 500101-1234567\n보호자 550202-2345678",
            "환자 500101-*******\n보호자 550202-*******",
        ),
    ],
)
def test_resident_number_back_digits_are_masked(text: str, expected: str) -> None:
    """주민등록번호·외국인등록번호는 구분자 모양과 상관없이 뒤 7자리를 가린다."""
    assert mask_resident_numbers(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "500101-1******",
        "500101-*******",
        "12345678901234",
        "010-1234-5678",
        "등록번호 12345678",
        "홍길동 2026-11-03 혈색소 13.5 g/dL",
    ],
)
def test_other_text_is_not_masked(text: str) -> None:
    """이미 가려진 번호, 더 긴 숫자, 전화번호, 이름, 날짜, 검사값은 그대로 둔다."""
    assert mask_resident_numbers(text) == text


def test_page_holds_masked_raw_text() -> None:
    """페이지 결과는 주민등록번호 뒷자리를 가린 원문만 담고, 이름은 그대로 둔다."""
    page = PageExtraction.model_validate(make_page(raw_text="홍길동 500101-1234567"))

    assert page.raw_text == "홍길동 500101-*******"


def test_json_schema_meets_strict_output_rules() -> None:
    """모든 객체가 전 필드 required, additionalProperties false다(OpenAI strict 조건)."""
    schema = PageExtraction.model_json_schema()
    objects = [schema, *schema["$defs"].values()]
    objects = [node for node in objects if node.get("type") == "object"]

    assert len(objects) == 3
    for node in objects:
        assert node["additionalProperties"] is False
        assert node["required"] == list(node["properties"])


def test_normalized_box_converts_to_pixels() -> None:
    """0~1000 좌표를 페이지 크기에 맞춰 픽셀로 바꾼다."""
    box = NormalizedBox(x=100, y=200, width=500, height=100)

    assert box.to_pixel_box(1240, 1754) == SourceBox(x=124, y=350, width=620, height=176)


@pytest.mark.parametrize(
    ("x", "y", "width", "height"),
    [(-1, 0, 10, 10), (0, 0, 0, 10), (0, 0, 10, -5), (900, 0, 200, 10), (0, 990, 10, 20)],
)
def test_box_outside_page_becomes_none(x: int, y: int, width: int, height: int) -> None:
    """음수 좌표, 크기 0 이하, 0~1000을 넘는 사각형은 None이다."""
    box = NormalizedBox(x=x, y=y, width=width, height=height)

    assert box.to_pixel_box(1240, 1754) is None
