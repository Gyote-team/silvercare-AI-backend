"""로컬 문서 저장소 테스트. pytest tmp_path 아래에 실제 파일을 만들어 읽고 쓴다."""

import os
from pathlib import Path

import pytest

from app.application.ports.document_storage import (
    DocumentNotFoundError,
    InvalidObjectKeyError,
    RenderedPageWriteError,
)
from app.infrastructure.storage.local_document_storage import LocalDocumentStorage

PAGE_KEY = "documents/1/pages/1.png"
KEYS_OUTSIDE_ROOT = [
    "../outside.txt",
    "documents/../../outside.txt",
    "",
    ".",
    "documents/..",
    "a\0b",
]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """저장소 루트와, 루트 밖에 실제로 존재하는 파일 하나를 만든다."""
    root = tmp_path / "root"
    (root / "documents" / "1").mkdir(parents=True)
    (root / "documents" / "1" / "original.pdf").write_bytes(b"original")
    (tmp_path / "outside.txt").write_bytes(b"outside")
    return root


def test_reads_original_bytes(root: Path) -> None:
    """루트 아래 objectKey 파일의 바이트를 그대로 읽는다."""
    storage = LocalDocumentStorage(str(root))

    assert storage.read_original("documents/1/original.pdf") == b"original"


@pytest.mark.parametrize("object_key", ["documents/1/missing.pdf", "documents/1"])
def test_missing_file_raises_not_found(root: Path, object_key: str) -> None:
    """파일이 없거나 폴더를 가리키면 DocumentNotFoundError다."""
    with pytest.raises(DocumentNotFoundError):
        LocalDocumentStorage(str(root)).read_original(object_key)


@pytest.mark.parametrize("object_key", KEYS_OUTSIDE_ROOT)
def test_key_outside_root_is_rejected(root: Path, object_key: str) -> None:
    """상위 경로, 루트 자체, 널 문자가 든 키는 InvalidObjectKeyError다."""
    with pytest.raises(InvalidObjectKeyError):
        LocalDocumentStorage(str(root)).read_original(object_key)


def test_absolute_path_key_is_rejected(root: Path) -> None:
    """루트 밖 실제 파일의 절대경로를 키로 줘도 읽지 않는다."""
    with pytest.raises(InvalidObjectKeyError):
        LocalDocumentStorage(str(root)).read_original(str(root.parent / "outside.txt"))


@pytest.mark.parametrize(
    ("object_key", "error_type"),
    [("missing.pdf", DocumentNotFoundError), ("../outside.txt", InvalidObjectKeyError)],
)
def test_error_message_has_key_but_no_root_path(
    root: Path, object_key: str, error_type: type[Exception]
) -> None:
    """예외 메시지에는 objectKey만 있고 저장소 루트 경로는 없다."""
    with pytest.raises(error_type) as error:
        LocalDocumentStorage(str(root)).read_original(object_key)

    assert object_key in str(error.value)
    assert str(root) not in str(error.value)


def test_writes_rendered_page(root: Path) -> None:
    """상위 폴더가 없어도 만들고, 받은 바이트를 그대로 쓴다. 임시 파일은 남기지 않는다."""
    LocalDocumentStorage(str(root)).write_rendered_page(PAGE_KEY, b"png")

    assert (root / PAGE_KEY).read_bytes() == b"png"
    assert os.listdir(root / "documents" / "1" / "pages") == ["1.png"]


def test_write_overwrites_same_key(root: Path) -> None:
    """같은 키로 다시 쓰면 새 바이트로 바뀌고 파일은 하나다."""
    storage = LocalDocumentStorage(str(root))
    storage.write_rendered_page(PAGE_KEY, b"old")

    storage.write_rendered_page(PAGE_KEY, b"new")

    assert (root / PAGE_KEY).read_bytes() == b"new"
    assert os.listdir(root / "documents" / "1" / "pages") == ["1.png"]


@pytest.mark.parametrize("object_key", KEYS_OUTSIDE_ROOT)
def test_write_rejects_key_outside_root(root: Path, object_key: str) -> None:
    """읽기와 같은 규칙으로 루트 밖 키를 거절하고, 루트 밖 파일을 건드리지 않는다."""
    with pytest.raises(InvalidObjectKeyError):
        LocalDocumentStorage(str(root)).write_rendered_page(object_key, b"png")

    assert (root.parent / "outside.txt").read_bytes() == b"outside"


def test_failed_write_keeps_old_file_and_hides_path(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """교체에 실패하면 옛 파일이 그대로 남고, 임시 파일은 없고, 예외에 루트 경로가 없다."""
    storage = LocalDocumentStorage(str(root))
    storage.write_rendered_page(PAGE_KEY, b"old")

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError(f"cannot replace {target}")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(RenderedPageWriteError) as error:
        storage.write_rendered_page(PAGE_KEY, b"new")

    assert (root / PAGE_KEY).read_bytes() == b"old"
    assert os.listdir(root / "documents" / "1" / "pages") == ["1.png"]
    assert PAGE_KEY in str(error.value)
    assert str(root) not in str(error.value)
