"""objectKey로 문서 원본을 읽고 렌더링 이미지를 쓰는 저장소 포트와 예외를 정의한다."""

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


class OriginalDocumentReadError(Exception):
    """원본 파일이 있지만 읽지 못했다. 메시지에는 objectKey만 담고 파일 경로는 담지 않는다."""

    def __init__(self, object_key: str) -> None:
        super().__init__(f"원본 파일을 읽지 못했습니다: {object_key}")
        self.object_key = object_key


class RenderedPageWriteError(Exception):
    """렌더링 이미지를 쓰지 못했다. 메시지에는 objectKey만 담고 파일 경로는 담지 않는다."""

    def __init__(self, object_key: str) -> None:
        super().__init__(f"렌더링 이미지를 저장하지 못했습니다: {object_key}")
        self.object_key = object_key


class OriginalDocumentReader(Protocol):
    """objectKey로 원본 바이트를 읽는 인터페이스."""

    def read_original(self, object_key: str) -> bytes:
        """원본 바이트를 반환한다.

        실패하면 DocumentNotFoundError, InvalidObjectKeyError, OriginalDocumentReadError 중
        하나를 던진다.
        """
        ...


class RenderedPageWriter(Protocol):
    """objectKey에 렌더링 이미지를 쓰는 인터페이스."""

    def write_rendered_page(self, object_key: str, png_bytes: bytes) -> None:
        """PNG 바이트를 쓴다. 같은 키가 있으면 덮어쓴다.

        실패하면 InvalidObjectKeyError나 RenderedPageWriteError를 던진다.
        """
        ...
