"""의료 판단 요청과 의료문서 설명 요청을 구분한다."""

import pytest

from app.contracts.ai_result_document import QuestionType
from app.domain.ai_result.safety_check import SafetyCheck


@pytest.mark.parametrize(
    "question",
    [
        "진단서 내용을 쉽게 설명해 주세요",
        "이 진단 기록에 적힌 내용을 알려 주세요",
    ],
)
def test_diagnosis_document_explanation_is_allowed(question: str) -> None:
    assert SafetyCheck().classify(question) is None


@pytest.mark.parametrize(
    "question",
    [
        "이 증상으로 진단해 줘",
        "제 병명을 진단 내려 줘",
        "약을 끊어도 돼?",
        "치료 방법을 골라 줘",
    ],
)
def test_medical_decision_request_is_blocked(question: str) -> None:
    assert SafetyCheck().classify(question) is QuestionType.MEDICAL_DECISION_REQUEST
