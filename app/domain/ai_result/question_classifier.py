"""가이드형 RAG 채팅의 질문 유형을 규칙으로 분류한다."""

from app.contracts.ai_result_document import QuestionType


class QuestionClassifier:
    """MVP에서는 설명 가능한 규칙을 사용하며, 이후 LLM JSON 분류기를 추가할 수 있다."""

    def classify(self, question: str, hint: QuestionType | None) -> QuestionType:
        if hint and hint not in {QuestionType.MEDICAL_DECISION_REQUEST, QuestionType.UNSUPPORTED}:
            return self.canonical(hint)
        text = question.lower()
        if any(word in text for word in ("지난", "이전", "비교", "달라", "변화")):
            return QuestionType.TEST_RESULT_COMPARISON
        if any(word in text for word in ("참고범위", "정상범위", "기준범위")):
            return QuestionType.LAB_REFERENCE_RANGE
        if any(word in text for word in ("약", "복용", "처방")):
            return QuestionType.MEDICATION_SUMMARY
        if any(word in text for word in ("재진", "일정", "검사 언제", "예약")):
            return QuestionType.FOLLOW_UP_SCHEDULE
        if any(word in text for word in ("근거", "원문", "몇 쪽", "출처")):
            return QuestionType.SOURCE_LOOKUP
        if any(word in text for word in ("설명", "쉽게", "소견서", "진단서", "문서")):
            return QuestionType.DOCUMENT_EASY_EXPLANATION
        return QuestionType.UNSUPPORTED

    @staticmethod
    def canonical(question_type: QuestionType) -> QuestionType:
        if question_type is QuestionType.LAB_RESULT_COMPARISON:
            return QuestionType.TEST_RESULT_COMPARISON
        return question_type

    @staticmethod
    def retrieval_strategy(question_type: QuestionType) -> str:
        if question_type in {QuestionType.TEST_RESULT_COMPARISON, QuestionType.LAB_REFERENCE_RANGE}:
            return "STRUCTURED_LAB"
        if question_type is QuestionType.MEDICATION_SUMMARY:
            return "STRUCTURED_MEDICATION"
        if question_type is QuestionType.FOLLOW_UP_SCHEDULE:
            return "STRUCTURED_SCHEDULE"
        if question_type is QuestionType.SOURCE_LOOKUP:
            return "SOURCE_LOOKUP"
        return "STRUCTURED_EVIDENCE"
