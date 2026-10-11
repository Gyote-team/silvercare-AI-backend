"""VLM이 페이지 한 장에서 추출한 결과 형식. 항목 필드는 extracted_items 컬럼과 같다."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

NORMALIZED_BOX_MAX = 1000


class DocumentType(StrEnum):
    """문서 종류. documents.document_type의 CHECK 제약 값과 같다."""

    LAB_RESULT = "LAB_RESULT"
    PRESCRIPTION = "PRESCRIPTION"
    DIAGNOSIS = "DIAGNOSIS"
    DISCHARGE_GUIDE = "DISCHARGE_GUIDE"
    UNKNOWN = "UNKNOWN"


class ExtractedItemType(StrEnum):
    """추출 항목 종류. extracted_items.item_type의 CHECK 제약 값과 같다."""

    LAB_VALUE = "LAB_VALUE"
    MEDICATION = "MEDICATION"
    SCHEDULE = "SCHEDULE"
    DIAGNOSIS_TEXT = "DIAGNOSIS_TEXT"
    CAUTION = "CAUTION"
    OTHER = "OTHER"


class _StrictModel(BaseModel):
    """모르는 필드를 거절하고 값을 바꿀 수 없는 모델.

    OpenAI strict 구조화 출력 조건(additionalProperties false)과 맞춘다.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceBox(_StrictModel):
    """원문 위치 사각형. 페이지 PNG의 픽셀 좌표이고 왼쪽 위가 (0, 0)이다."""

    x: int
    y: int
    width: int
    height: int


class NormalizedBox(_StrictModel):
    """VLM이 돌려주는 원문 위치 사각형. 페이지 가로·세로를 각각 0~1000으로 본 좌표다."""

    x: int
    y: int
    width: int
    height: int

    def to_pixel_box(self, width_px: int, height_px: int) -> SourceBox | None:
        """페이지 픽셀 좌표로 바꾼다. 페이지를 벗어나거나 크기가 없으면 None을 반환한다."""
        right = self.x + self.width
        bottom = self.y + self.height
        if min(self.x, self.y) < 0 or min(self.width, self.height) <= 0:
            return None
        if max(right, bottom) > NORMALIZED_BOX_MAX:
            return None
        left_px = self.x * width_px // NORMALIZED_BOX_MAX
        top_px = self.y * height_px // NORMALIZED_BOX_MAX
        return SourceBox(
            x=left_px,
            y=top_px,
            width=max(1, right * width_px // NORMALIZED_BOX_MAX - left_px),
            height=max(1, bottom * height_px // NORMALIZED_BOX_MAX - top_px),
        )


class ExtractedItem(_StrictModel):
    """문서에서 옮겨 적은 항목 하나. 기본값이 없어 null이어도 키는 모두 있어야 한다."""

    item_type: ExtractedItemType
    normalized_name: str | None = Field(max_length=200)
    raw_value: str | None = Field(max_length=500)
    normalized_value: str | None = Field(max_length=500)
    unit: str | None = Field(max_length=50)
    reference_range: str | None = Field(max_length=100)
    confidence: float | None = Field(ge=0, le=1)
    source_text: str
    source_box: SourceBox | None

    @field_validator("source_text")
    @classmethod
    def _require_source_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source_text가 비어 있습니다")
        return value


class PageExtraction(_StrictModel):
    """페이지 한 장의 추출 결과. 옮길 항목이 없으면 items는 빈 목록이다."""

    document_type: DocumentType
    items: list[ExtractedItem]
