"""VLM 포트를 감싸 일시적인 실패만 다시 부르는 추출기."""

import time
from collections.abc import Callable

from app.application.ports.page_extractor import (
    PageExtractionError,
    PageExtractionErrorCode,
    PageExtractionResult,
    PageExtractor,
)
from app.domain.document.page_split import PageImage

# 다시 부르면 성공할 수 있는 오류. 나머지는 같은 요청을 다시 보내도 결과가 같다.
_RETRYABLE_CODES = frozenset(
    {
        PageExtractionErrorCode.TIMEOUT,
        PageExtractionErrorCode.RATE_LIMITED,
        PageExtractionErrorCode.API_ERROR,
    }
)


class RetryingPageExtractor:
    """안쪽 추출기가 재시도 대상 오류로 실패하면 기다렸다가 다시 부른다.

    대기 시간은 backoff_seconds에서 시작해 재시도마다 두 배가 된다.
    """

    def __init__(
        self,
        inner: PageExtractor,
        *,
        max_retries: int,
        backoff_seconds: float,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._inner = inner
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep

    def extract_page(self, page: PageImage) -> PageExtractionResult:
        """처음 호출 포함 최대 max_retries + 1번 부른다. 끝까지 실패하면 마지막 오류를 올린다."""
        attempt = 0
        while True:
            try:
                return self._inner.extract_page(page)
            except PageExtractionError as error:
                if error.code not in _RETRYABLE_CODES or attempt >= self._max_retries:
                    raise
            self._sleep(self._backoff_seconds * 2**attempt)
            attempt += 1
