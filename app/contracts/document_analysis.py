"""Spring이 문서 업로드 후 보내는 분석 요청의 내부 API 계약이다."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class DocumentAnalysisRequest(BaseModel):
    """분석 시작 요청 본문이다. JSON 필드는 Spring 계약대로 camelCase다.

    objectKey·mimeType은 형식만 검증한다. 분석은 DB의 storage_key·mime_type을 사용한다.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    document_id: UUID
    object_key: str = Field(min_length=1)
    mime_type: str = Field(min_length=1)
    # Spring은 UUID가 아닌 문자열(예: upload-20261003-abcd1234)을 보낸다.
    # 길이는 Spring request_id 컬럼(VARCHAR(64))과 같다. 로그에 남기므로 개행 같은 문자는 막는다.
    request_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
