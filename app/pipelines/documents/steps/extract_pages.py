"""페이지마다 VLM을 한 번씩 불러 추출 결과를 모으는 문서 파이프라인 단계."""

from collections.abc import Sequence
from dataclasses import dataclass

from app.application.ports.page_extractor import (
    PageExtractionError,
    PageExtractionErrorCode,
    PageExtractionResult,
    PageExtractor,
)
from app.domain.document.page_split import PageImage


@dataclass(frozen=True)
class PageExtractionOutcome:
    """페이지 한 장의 추출 성공 여부. 성공이면 result, 실패면 error_code 하나만 값이 있다."""

    page_no: int
    result: PageExtractionResult | None = None
    error_code: PageExtractionErrorCode | None = None


def extract_pages(
    extractor: PageExtractor, pages: Sequence[PageImage]
) -> list[PageExtractionOutcome]:
    """페이지 순서대로 VLM을 부른다. 실패한 페이지는 오류 코드로 남기고 다음 페이지를 계속한다."""
    outcomes: list[PageExtractionOutcome] = []
    # 페이지를 하나씩 차례로 호출한다. 문서 한 건이 너무 느리면 스레드 풀로
    # 여러 페이지를 동시에 호출하도록 바꿀 수 있다.
    for page in pages:
        try:
            result = extractor.extract_page(page)
        except PageExtractionError as error:
            outcomes.append(PageExtractionOutcome(page_no=page.page_no, error_code=error.code))
        else:
            outcomes.append(PageExtractionOutcome(page_no=page.page_no, result=result))
    return outcomes
