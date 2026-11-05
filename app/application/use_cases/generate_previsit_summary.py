"""모델이 반환한 모든 근거가 허용 목록에 있는지 검증한 뒤 요약을 반환한다."""

from app.application.ports.previsit_summarizer import PrevisitFailure, PrevisitSummarizer
from app.contracts.previsit_summary import PrevisitSummaryRequest, PrevisitSummaryResponse


class GeneratePrevisitSummaryUseCase:
    """다른 환자의 기록·검색·문서 분석 결과는 사용하지 않는 요약 유스케이스다."""

    def __init__(self, provider: PrevisitSummarizer) -> None:
        self.provider = provider

    def execute(self, request: PrevisitSummaryRequest) -> PrevisitSummaryResponse:
        """기록 없음은 근거 부족, 잘못된 생성 결과는 노출 없이 실패로 반환한다."""
        if not request.records:
            return PrevisitSummaryResponse(
                request_id=request.request_id,
                summary="선택한 기간에 작성된 건강기록이 없어요.",
                result_status="INSUFFICIENT",
            )
        try:
            generated = self.provider.generate(request)
            allowed = {r.health_record_id for r in request.records}
            used = []
            texts = []
            for sentence in generated.sentences:
                if (
                    not sentence.text.strip()
                    or not set(sentence.source_health_record_ids) <= allowed
                ):
                    raise PrevisitFailure("AI_RESPONSE_INVALID")
                texts.append(sentence.text.strip())
                for identifier in sentence.source_health_record_ids:
                    if identifier not in used:
                        used.append(identifier)
            summary = "\n".join(texts)
            if not used or not summary or len(summary) > 12000:
                raise PrevisitFailure("AI_RESPONSE_INVALID")
            return PrevisitSummaryResponse(
                request_id=request.request_id,
                summary=summary,
                used_health_record_ids=used,
                result_status="COMPLETE",
            )
        except PrevisitFailure as error:
            return PrevisitSummaryResponse(
                request_id=request.request_id,
                failed=True,
                error_code=error.code,
                retryable=error.retryable,
            )
