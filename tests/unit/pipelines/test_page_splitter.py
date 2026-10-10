"""페이지 분리 단위 테스트. 입력 파일은 모두 테스트 안에서 합성한다."""

import struct
import zlib
from io import BytesIO

import pymupdf
import pytest
from PIL import Image

from app.domain.document.page_split import PageImage, PageSplitError, PageSplitErrorCode
from app.pipelines.documents.page_splitter import split_pages

PDF = "application/pdf"
JPEG = "image/jpeg"
PNG = "image/png"
PAGE_SIZES = ((300, 400), (400, 300), (200, 200))
ZERO_PAGE_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
    b"trailer\n<< /Root 1 0 R >>\n%%EOF\n"
)


def make_pdf(**save_options: object) -> bytes:
    """PAGE_SIZES 크기의 빈 페이지 3장짜리 PDF를 만든다."""
    document = pymupdf.open()
    for width, height in PAGE_SIZES:
        document.new_page(width=width, height=height)
    return document.tobytes(**save_options)


def make_image(image_format: str, *, mode: str = "RGB", orientation: int | None = None) -> bytes:
    """가로 40, 세로 20 이미지를 만든다. orientation을 주면 EXIF 회전 값을 넣는다."""
    image = Image.new(mode, (40, 20))
    options = {}
    if orientation is not None:
        exif = image.getexif()
        exif[0x0112] = orientation
        options["exif"] = exif
    buffer = BytesIO()
    image.save(buffer, format=image_format, **options)
    return buffer.getvalue()


def make_broken_chunk_png() -> bytes:
    """서명은 맞지만 IHDR 청크 길이가 모자란 PNG를 만든다."""
    body = b"\x00" * 5
    crc = struct.pack(">I", zlib.crc32(b"IHDR" + body))
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", len(body)) + b"IHDR" + body + crc


def split(
    data: bytes, mime_type: str, *, render_dpi: int = 72, max_pages: int = 30
) -> list[PageImage]:
    """테스트 기본값(72 DPI, 최대 30페이지)으로 split_pages를 부른다."""
    return split_pages(data, mime_type, render_dpi=render_dpi, max_pages=max_pages)


def assert_split_fails(
    data: bytes, mime_type: str, code: PageSplitErrorCode, *, max_pages: int = 30
) -> None:
    """분리가 주어진 오류 코드로 실패하는지 확인한다."""
    with pytest.raises(PageSplitError) as error:
        split(data, mime_type, max_pages=max_pages)
    assert error.value.code is code


def test_pdf_pages_are_numbered_from_one_with_sizes() -> None:
    """여러 페이지 PDF는 1부터 번호가 붙고 페이지마다 크기가 기록된다."""
    pages = split(make_pdf(), PDF)

    assert [page.page_no for page in pages] == [1, 2, 3]
    assert [(page.width_px, page.height_px) for page in pages] == list(PAGE_SIZES)
    for page in pages:
        with Image.open(BytesIO(page.png_bytes)) as image:
            assert image.format == "PNG"
            assert image.size == (page.width_px, page.height_px)


def test_pdf_render_dpi_scales_page_size() -> None:
    """DPI를 두 배로 하면 픽셀 크기도 두 배가 된다."""
    pages = split(make_pdf(), PDF, render_dpi=144)

    assert (pages[0].width_px, pages[0].height_px) == (600, 800)


@pytest.mark.parametrize(("mime_type", "image_format"), [(PNG, "PNG"), (JPEG, "JPEG")])
def test_image_becomes_single_png_page(mime_type: str, image_format: str) -> None:
    """이미지 한 장은 1번 페이지 PNG 하나가 된다."""
    pages = split(make_image(image_format), mime_type)

    assert len(pages) == 1
    assert (pages[0].page_no, pages[0].width_px, pages[0].height_px) == (1, 40, 20)
    with Image.open(BytesIO(pages[0].png_bytes)) as image:
        assert image.format == "PNG"
        assert image.size == (40, 20)


def test_exif_rotation_is_applied() -> None:
    """EXIF 회전 값이 있으면 돌린 뒤의 크기를 기록한다."""
    pages = split(make_image("JPEG", orientation=6), JPEG)

    assert (pages[0].width_px, pages[0].height_px) == (20, 40)
    with Image.open(BytesIO(pages[0].png_bytes)) as image:
        assert image.size == (20, 40)


def test_cmyk_jpeg_is_converted_to_png() -> None:
    """PNG가 담지 못하는 CMYK 이미지도 PNG로 변환된다."""
    pages = split(make_image("JPEG", mode="CMYK"), JPEG)

    with Image.open(BytesIO(pages[0].png_bytes)) as image:
        assert image.format == "PNG"


@pytest.mark.parametrize(
    ("mime_type", "data"),
    [
        (PDF, b"not a pdf"),
        (PDF, b""),
        (JPEG, b"not a jpeg"),
        (PNG, b"not a png"),
        (PNG, make_broken_chunk_png()),
    ],
)
def test_corrupted_bytes_are_rejected(mime_type: str, data: bytes) -> None:
    """손상됐거나 열 수 없는 바이트는 CORRUPTED_FILE이다."""
    assert_split_fails(data, mime_type, PageSplitErrorCode.CORRUPTED_FILE)


@pytest.mark.parametrize(
    ("mime_type", "data"),
    [(PDF, make_image("PNG")), (PNG, make_image("JPEG")), (JPEG, make_image("PNG"))],
)
def test_mime_type_mismatch_is_rejected(mime_type: str, data: bytes) -> None:
    """선언한 mimeType과 실제 형식이 다르면 CORRUPTED_FILE이다."""
    assert_split_fails(data, mime_type, PageSplitErrorCode.CORRUPTED_FILE)


def test_zero_page_pdf_is_rejected() -> None:
    """페이지가 없는 PDF는 EMPTY_DOCUMENT다."""
    assert_split_fails(ZERO_PAGE_PDF, PDF, PageSplitErrorCode.EMPTY_DOCUMENT)


def test_open_password_pdf_is_rejected() -> None:
    """열기 비밀번호가 있는 PDF는 PASSWORD_PROTECTED다."""
    data = make_pdf(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="open", owner_pw="owner")

    assert_split_fails(data, PDF, PageSplitErrorCode.PASSWORD_PROTECTED)


def test_permission_restricted_pdf_is_processed() -> None:
    """소유자 비밀번호로 인쇄·복사만 막은 PDF는 정상 처리된다."""
    data = make_pdf(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        permissions=pymupdf.PDF_PERM_ACCESSIBILITY,
    )
    with pymupdf.open(stream=data, filetype="pdf") as document:
        assert document.permissions & (pymupdf.PDF_PERM_PRINT | pymupdf.PDF_PERM_COPY) == 0

    assert [page.page_no for page in split(data, PDF)] == [1, 2, 3]


def test_page_limit_is_enforced() -> None:
    """최대 페이지 수와 같으면 처리하고 넘으면 PAGE_LIMIT_EXCEEDED다."""
    assert len(split(make_pdf(), PDF, max_pages=3)) == 3
    assert_split_fails(make_pdf(), PDF, PageSplitErrorCode.PAGE_LIMIT_EXCEEDED, max_pages=2)


@pytest.mark.parametrize("mime_type", ["image/gif", "text/plain", ""])
def test_unsupported_mime_type_is_rejected(mime_type: str) -> None:
    """허용하지 않는 mimeType은 UNSUPPORTED_MIME_TYPE이다."""
    assert_split_fails(make_pdf(), mime_type, PageSplitErrorCode.UNSUPPORTED_MIME_TYPE)
