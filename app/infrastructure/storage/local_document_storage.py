"""로컬 파일시스템에서 문서 원본을 읽는 저장소 구현."""

import os
from pathlib import Path

from app.application.ports.document_storage import DocumentNotFoundError, InvalidObjectKeyError


class LocalDocumentStorage:
    """루트 폴더 아래에서만 원본을 읽는다.

    경로 규칙은 Spring LocalDocumentStorage와 같다.
    """

    def __init__(self, root: str) -> None:
        self._root = Path(os.path.abspath(root))

    def read_original(self, object_key: str) -> bytes:
        """루트 아래 objectKey 파일의 바이트를 반환한다."""
        target = self._resolve_inside_root(object_key)
        if not target.is_file():
            raise DocumentNotFoundError(object_key)
        return target.read_bytes()

    def _resolve_inside_root(self, object_key: str) -> Path:
        if "\0" in object_key:
            raise InvalidObjectKeyError(object_key)
        target = Path(os.path.abspath(os.path.join(self._root, object_key)))
        if target == self._root or not target.is_relative_to(self._root):
            raise InvalidObjectKeyError(object_key)
        return target
