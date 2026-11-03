"""진료 전 요약 공급자의 교체 가능한 경계다."""

from typing import Protocol

from app.contracts.previsit_summary import GeneratedPrevisitSummary, PrevisitSummaryRequest


class PrevisitFailure(Exception):
    """원문 없이 안전한 실패 코드와 재시도 가능 여부만 담는다."""

    def __init__(self, code: str, retryable: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class PrevisitSummarizer(Protocol):
    """허용된 건강기록에서 근거 연결 문장을 생성한다."""

    def generate(self, request: PrevisitSummaryRequest) -> GeneratedPrevisitSummary:
        """의료적 판단 없이 입력 기록을 요약한다."""
        ...
