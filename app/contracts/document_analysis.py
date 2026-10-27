"""Spring이 문서 업로드 후 보내는 분석 요청의 내부 API 계약이다."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class DocumentAnalysisRequest(BaseModel):
    """분석 시작 요청 본문이다. JSON 필드는 Spring 계약대로 camelCase다."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    document_id: UUID
    object_key: str = Field(min_length=1)
    mime_type: str = Field(min_length=1)
    # Spring은 UUID가 아닌 문자열(예: upload-20261003-abcd1234)을 보낸다.
    request_id: str = Field(min_length=1)
