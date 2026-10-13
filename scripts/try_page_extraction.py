"""합성 문서 한 장으로 실제 OpenAI 추출을 한 번 호출해 보는 수동 확인 스크립트.

자동 테스트에 포함되지 않는다. 실행할 때마다 OpenAI 비용이 든다.
실제 진료 문서를 넣지 않는다. 저장소 루트에서 실행한다.

    python -m scripts.try_page_extraction
"""

import time

import pymupdf

from app.application.ports.page_extractor import PageExtractionError
from app.core.config import get_settings
from app.infrastructure.providers.llm.openai_page_extractor import create_page_extractor
from app.pipelines.documents.page_splitter import split_pages

_SYNTHETIC_LINES = [
    "합성 검사 결과지 (테스트용, 실제 환자 정보 아님)",
    "검사일: 2026-10-01",
    "혈색소(Hb)   13.5 g/dL   (참고치 12.0-16.0)",
    "공복혈당   132 mg/dL   H   (참고치 70-99)",
    "총콜레스테롤   210 mg/dL   H   (참고치 0-199)",
    "다음 진료: 2026-11-03 오전 10시 내과",
    "주의: 검사 전 8시간 금식",
]


def make_synthetic_pdf() -> bytes:
    """지어낸 검사 결과 몇 줄이 적힌 A4 한 장짜리 PDF를 만든다."""
    document = pymupdf.open()
    page = document.new_page()
    for index, line in enumerate(_SYNTHETIC_LINES):
        # "korea"는 PyMuPDF에 들어 있는 한글 글꼴이다.
        page.insert_text((72, 90 + index * 30), line, fontsize=13, fontname="korea")
    return document.tobytes()


def main() -> None:
    """합성 페이지 한 장을 한 번 호출하고 걸린 시간, 사용량, 추출 결과를 출력한다."""
    settings = get_settings()
    pages = split_pages(
        make_synthetic_pdf(),
        "application/pdf",
        render_dpi=settings.document_render_dpi,
        max_pages=settings.document_max_pages,
        max_side_px=settings.document_max_page_side_px,
    )
    extractor = create_page_extractor(settings)

    started = time.perf_counter()
    try:
        result = extractor.extract_page(pages[0])
    except PageExtractionError as error:
        print(f"실패: {error.code.value} ({time.perf_counter() - started:.1f}초)")
        raise SystemExit(1) from None

    print(f"걸린 시간: {time.perf_counter() - started:.1f}초")
    print(f"모델: {result.usage.model_name}, 프롬프트: {result.usage.prompt_version}")
    print(
        f"토큰: 입력 {result.usage.input_tokens}, 출력 {result.usage.output_tokens}"
        f" (추론 {result.usage.reasoning_tokens})"
    )
    # 합성 문서라 결과를 그대로 출력한다. 실제 문서의 추출 결과는 출력하거나 로그에 남기지 않는다.
    print(result.extraction.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
