"""원본 파일을 페이지로 나눈 결과와 실패 이유를 정의한다."""

from dataclasses import dataclass, field
from enum import Enum


class PageSplitErrorCode(Enum):
    """원본 파일을 페이지로 나누지 못한 이유."""

    CORRUPTED_FILE = "CORRUPTED_FILE"  # 손상·열 수 없음
    EMPTY_DOCUMENT = "EMPTY_DOCUMENT"  # 0페이지
    PASSWORD_PROTECTED = "PASSWORD_PROTECTED"  # 열기 비밀번호가 있는 PDF
    PAGE_LIMIT_EXCEEDED = "PAGE_LIMIT_EXCEEDED"  # 최대 페이지 수 초과
    UNSUPPORTED_MIME_TYPE = "UNSUPPORTED_MIME_TYPE"  # 지원하지 않는 형식


class PageSplitError(Exception):
    """페이지 분리 실패. 메시지에는 오류 코드만 담고 파일 내용은 담지 않는다."""

    def __init__(self, code: PageSplitErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


@dataclass(frozen=True)
class PageImage:
    """PNG로 렌더링한 페이지 한 장. page_no는 1부터 시작한다."""

    page_no: int
    png_bytes: bytes = field(repr=False)
    width_px: int
    height_px: int
