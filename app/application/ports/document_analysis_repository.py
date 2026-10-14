"""문서 분석의 대상 조회, 상태 기록, 추출 결과 저장 포트를 정의한다."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from app.domain.document.document_analysis import AnalysisFailure, DocumentStatus
from app.domain.document.page_extraction import PageExtraction


@dataclass(frozen=True)
class AnalysisTarget:
    """분석할 문서와 그 문서의 최신 분석 작업."""

    document_id: UUID
    analysis_id: UUID
    storage_key: str
    mime_type: str


class DocumentAnalysisRepository(Protocol):
    """분석 상태와 추출 결과를 저장하는 인터페이스. 메서드 하나가 트랜잭션 하나다.

    상태를 바꾸는 메서드는 문서가 삭제됐으면 아무것도 바꾸지 않고 False를 반환한다.
    """

    def find_target(self, document_id: UUID) -> AnalysisTarget | None:
        """문서와 최신 분석 작업을 읽는다. 문서가 없거나 삭제됐거나 분석 작업이 없으면 None이다."""
        ...

    def mark_started(self, target: AnalysisTarget, started_at: datetime) -> bool:
        """문서를 PROCESSING, 분석을 RUNNING으로 바꾸고 이전 실패 기록을 지운다.

        분석이 이미 SUCCEEDED면 아무것도 바꾸지 않고 False를 반환한다. 재분석은 같은 행이 아니라
        새 분석 행으로 한다(시트 FC-08-03-02, 재시도 API가 새 행을 만든다).
        """
        ...

    def save_result(
        self,
        target: AnalysisTarget,
        document_status: DocumentStatus,
        extractions_by_page_no: Mapping[int, PageExtraction],
        model_version: str | None,
        completed_at: datetime,
    ) -> bool:
        """이 분석의 추출 항목을 새 항목으로 바꾸고, 분석을 SUCCEEDED, 문서를 받은 상태로 바꾼다.

        extractions_by_page_no에는 추출에 성공한 페이지만 담는다. 그 페이지는 원문과 함께
        SUCCEEDED(글자가 없으면 원문 없이 SKIPPED)로, 나머지 페이지는 원문 없이 FAILED로 기록한다.
        """
        ...

    def mark_failed(
        self, target: AnalysisTarget, failure: AnalysisFailure, completed_at: datetime
    ) -> bool:
        """이 분석의 추출 항목을 지우고, 문서와 분석을 FAILED로 바꾸고 실패 기록을 남긴다.

        문서의 모든 페이지도 원문 없이 FAILED로 바꾼다. 설명이 참조해 지울 수 없는 항목은 남긴다.
        """
        ...
