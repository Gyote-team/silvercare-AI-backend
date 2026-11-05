"""Spring이 권한 검증한 건강기록과 진료 전 요약의 내부 계약이다."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel


class PrevisitSource(BaseModel):
    """요약에 사용할 수 있는 단일 건강기록 스냅샷이다."""

    model_config = ConfigDict(extra="forbid", alias_generator=to_camel, populate_by_name=True)
    health_record_id: UUID
    recorded_at: datetime
    content: str = Field(min_length=1, max_length=2000)


class PrevisitSummaryRequest(BaseModel):
    """대상 선정과 권한 검증은 Spring이 수행하며 AI는 다른 기록을 검색하지 않는다."""

    model_config = ConfigDict(extra="forbid", alias_generator=to_camel, populate_by_name=True)
    request_id: UUID
    window_days: int = Field(ge=1, le=30)
    records: list[PrevisitSource] = Field(max_length=100)

    @model_validator(mode="after")
    def validate_sources(self) -> "PrevisitSummaryRequest":
        """중복 근거와 과도한 입력은 자동 절단하지 않고 거절한다."""
        ids = [r.health_record_id for r in self.records]
        if len(set(ids)) != len(ids) or sum(len(r.content) for r in self.records) > 60000:
            raise ValueError("Invalid source collection")
        return self


class PrevisitSentence(BaseModel):
    """각 요약 문장은 제공된 건강기록을 하나 이상 참조해야 한다."""

    model_config = ConfigDict(extra="forbid", alias_generator=to_camel, populate_by_name=True)
    text: str = Field(min_length=1, max_length=2000)
    source_health_record_ids: list[UUID] = Field(min_length=1, max_length=100)


class GeneratedPrevisitSummary(BaseModel):
    """모델이 반환하는 근거 연결 문장 집합이다."""

    model_config = ConfigDict(extra="forbid", alias_generator=to_camel, populate_by_name=True)
    sentences: list[PrevisitSentence] = Field(min_length=1, max_length=30)


class PrevisitSummaryResponse(BaseModel):
    """오류 원문을 포함하지 않는 요약 성공·실패 응답이다."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    request_id: UUID
    summary: str | None = None
    used_health_record_ids: list[UUID] = Field(default_factory=list)
    result_status: Literal["COMPLETE", "INSUFFICIENT"] | None = None
    failed: bool = False
    error_code: str | None = None
    retryable: bool = False
