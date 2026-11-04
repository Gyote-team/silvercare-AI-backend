from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.contracts.ai_result_document import EvidenceSource, GenerateResultDocumentRequest


def test_rejects_document_evidence_outside_allowed_scope():
    allowed_document_id = uuid4()
    other_document_id = uuid4()

    with pytest.raises(ValidationError, match="허용 범위 밖의 문서 근거"):
        GenerateResultDocumentRequest(
            request_id=uuid4(),
            patient_id=uuid4(),
            actor_user_id=uuid4(),
            question="문서를 설명해 줘",
            allowed_document_ids=[allowed_document_id],
            evidence=[
                EvidenceSource(
                    citation_id=uuid4(),
                    source_type="DOCUMENT_CHUNK",
                    document_id=other_document_id,
                    source_text="검사 결과",
                )
            ],
        )


def test_maps_health_record_source_id_and_accepts_allowed_scope():
    health_record_id = uuid4()

    request = GenerateResultDocumentRequest(
        request_id=uuid4(),
        patient_id=uuid4(),
        actor_user_id=uuid4(),
        question="건강기록을 설명해 줘",
        allowed_health_record_ids=[health_record_id],
        evidence=[
            EvidenceSource(
                citation_id=uuid4(),
                source_type="HEALTH_RECORD",
                source_id=health_record_id,
                source_text="어지러움 기록",
            )
        ],
    )

    assert request.evidence[0].health_record_id == health_record_id
