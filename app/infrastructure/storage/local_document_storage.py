"""로컬 파일시스템에서 문서 원본을 읽고 렌더링 이미지를 쓰는 저장소 구현."""

import os
from pathlib import Path
from uuid import uuid4

from app.application.ports.document_storage import (
    DocumentNotFoundError,
    InvalidObjectKeyError,
    OriginalDocumentReadError,
    RenderedPageWriteError,
)


class LocalDocumentStorage:
    """루트 폴더 아래에서만 원본을 읽고 렌더링 이미지를 쓴다.

    경로 규칙은 Spring LocalDocumentStorage와 같다.
    """

    def __init__(self, root: str) -> None:
        self._root = Path(os.path.abspath(root))

    def read_original(self, object_key: str) -> bytes:
        """루트 아래 objectKey 파일의 바이트를 반환한다."""
        target = self._resolve_inside_root(object_key)
        if not target.is_file():
            raise DocumentNotFoundError(object_key)
        # OSError 메시지에는 전체 파일 경로가 들어 있으므로 from None으로 끊는다.
        try:
            return target.read_bytes()
        except OSError:
            raise OriginalDocumentReadError(object_key) from None

    def write_rendered_page(self, object_key: str, png_bytes: bytes) -> None:
        """루트 아래 objectKey에 PNG를 쓴다. 같은 키가 있으면 덮어쓴다."""
        target = self._resolve_inside_root(object_key)
        # 쓰다가 실패해도 깨진 파일이 남지 않게 같은 폴더의 임시 파일에 쓴 뒤 교체한다.
        temporary = target.with_name(f"{target.name}.{uuid4().hex}.tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(png_bytes)
            os.replace(temporary, target)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise RenderedPageWriteError(object_key) from None

    def _resolve_inside_root(self, object_key: str) -> Path:
        if "\0" in object_key:
            raise InvalidObjectKeyError(object_key)
        target = Path(os.path.abspath(os.path.join(self._root, object_key)))
        if target == self._root or not target.is_relative_to(self._root):
            raise InvalidObjectKeyError(object_key)
        return target
