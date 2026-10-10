"""objectKey로 문서 원본을 읽는 저장소 포트와 읽기 예외를 정의한다."""

from typing import Protocol


class DocumentNotFoundError(Exception):
    """objectKey에 해당하는 원본 파일이 없다."""

    def __init__(self, object_key: str) -> None:
        super().__init__(f"원본 파일이 없습니다: {object_key}")
        self.object_key = object_key


class InvalidObjectKeyError(Exception):
    """objectKey가 저장소 루트 밖이나 루트 자체를 가리키거나 형식이 올바르지 않다."""

    def __init__(self, object_key: str) -> None:
        super().__init__(f"objectKey가 올바르지 않습니다: {object_key}")
        self.object_key = object_key


class OriginalDocumentReader(Protocol):
    """objectKey로 원본 바이트를 읽는 인터페이스."""

    def read_original(self, object_key: str) -> bytes:
        """원본 바이트를 반환한다. 실패하면 이 모듈의 두 예외 중 하나를 던진다."""
        ...
