"""objectKey로 원본을 읽어 페이지로 나누는 문서 파이프라인 단계."""

from app.application.ports.document_storage import OriginalDocumentReader
from app.domain.document.page_split import PageImage
from app.pipelines.documents.page_splitter import split_pages


def load_document_pages(
    reader: OriginalDocumentReader,
    object_key: str,
    mime_type: str,
    *,
    render_dpi: int,
    max_pages: int,
) -> list[PageImage]:
    """objectKey의 원본을 읽어 페이지별 PNG 목록으로 반환한다."""
    data = reader.read_original(object_key)
    return split_pages(data, mime_type, render_dpi=render_dpi, max_pages=max_pages)
