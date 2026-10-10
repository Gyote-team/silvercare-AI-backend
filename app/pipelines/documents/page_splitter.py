"""원본 바이트를 페이지별 PNG로 나눈다. PyMuPDF와 Pillow는 이 모듈에서만 쓴다."""

from io import BytesIO

import pymupdf
from PIL import Image, ImageOps

from app.domain.document.page_split import PageImage, PageSplitError, PageSplitErrorCode

# MuPDF가 손상 파일 복구 메시지를 표준 출력에 직접 쓰지 않게 한다.
pymupdf.TOOLS.mupdf_display_errors(False)

_PDF_MIME_TYPE = "application/pdf"
_IMAGE_FORMATS = {"image/jpeg": "JPEG", "image/png": "PNG"}
_PNG_MODES = ("1", "L", "LA", "P", "RGB", "RGBA")


def split_pages(data: bytes, mime_type: str, *, render_dpi: int, max_pages: int) -> list[PageImage]:
    """원본 바이트를 페이지별 PNG로 나눈다. 실패하면 PageSplitError를 던진다."""
    if mime_type == _PDF_MIME_TYPE:
        return _split_pdf(data, render_dpi, max_pages)
    if mime_type in _IMAGE_FORMATS:
        return [_convert_image(data, _IMAGE_FORMATS[mime_type])]
    raise PageSplitError(PageSplitErrorCode.UNSUPPORTED_MIME_TYPE)


def _split_pdf(data: bytes, render_dpi: int, max_pages: int) -> list[PageImage]:
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except pymupdf.FileDataError:
        raise PageSplitError(PageSplitErrorCode.CORRUPTED_FILE) from None
    with document:
        # filetype="pdf"여도 이미지 바이트는 이미지 문서로 열리므로 따로 막는다.
        if not document.is_pdf:
            raise PageSplitError(PageSplitErrorCode.CORRUPTED_FILE)
        # 소유자 비밀번호만 있는 PDF는 needs_pass가 거짓이라 그대로 렌더링된다.
        if document.needs_pass:
            raise PageSplitError(PageSplitErrorCode.PASSWORD_PROTECTED)
        if document.page_count == 0:
            raise PageSplitError(PageSplitErrorCode.EMPTY_DOCUMENT)
        if document.page_count > max_pages:
            raise PageSplitError(PageSplitErrorCode.PAGE_LIMIT_EXCEEDED)
        try:
            return [
                _render_page(page, page_no, render_dpi)
                for page_no, page in enumerate(document, start=1)
            ]
        except Exception:  # MuPDF 렌더링 예외는 공통 상위 타입이 Exception뿐이다.
            raise PageSplitError(PageSplitErrorCode.CORRUPTED_FILE) from None


def _render_page(page: pymupdf.Page, page_no: int, render_dpi: int) -> PageImage:
    pixmap = page.get_pixmap(dpi=render_dpi, alpha=False)
    return PageImage(
        page_no=page_no,
        png_bytes=pixmap.tobytes("png"),
        width_px=pixmap.width,
        height_px=pixmap.height,
    )


def _convert_image(data: bytes, image_format: str) -> PageImage:
    try:
        with Image.open(BytesIO(data), formats=[image_format]) as source:
            image = ImageOps.exif_transpose(source)
            if image.mode not in _PNG_MODES:
                image = image.convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="PNG")
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise PageSplitError(PageSplitErrorCode.CORRUPTED_FILE) from None
    return PageImage(
        page_no=1,
        png_bytes=buffer.getvalue(),
        width_px=image.width,
        height_px=image.height,
    )
