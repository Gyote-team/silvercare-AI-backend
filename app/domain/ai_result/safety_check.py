"""의료 판단·처방 변경 질문을 보수적으로 차단한다."""

from app.contracts.ai_result_document import QuestionType


class SafetyCheck:
    """규칙 기반 1차 안전 검사다. LLM은 차단 뒤 호출하지 않는다."""

    _medical_decision_keywords = (
        "진단해",
        "진단을 내려",
        "진단 내려",
        "병인가",
        "병이야",
        "약을 끊",
        "약 끊",
        "약을 바꿔",
        "약 바꿔",
        "용량을 늘",
        "용량 늘",
        "용량을 줄",
        "치료 방법",
        "치료를 선택",
        "응급실",
        "응급인가",
    )

    def classify(self, question: str) -> QuestionType | None:
        """의료 판단으로 보이면 즉시 차단 유형을 반환한다."""
        normalized = question.replace(" ", "").lower()
        has_medical_decision_keyword = any(
            keyword.replace(" ", "") in normalized
            for keyword in self._medical_decision_keywords
        )
        if has_medical_decision_keyword:
            return QuestionType.MEDICAL_DECISION_REQUEST
        return None

    @staticmethod
    def user_message() -> str:
        return (
            "의료기록 내용을 정리할 수는 있지만 진단·약 변경·치료 결정은 "
            "의료진과 확인해 주세요."
        )
