"""영속 음성 큐의 저장소 구현과 변환 유스케이스를 분리한다."""

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True)
class VoiceJob:
    """허용된 파일과 실행 lease를 고정하는 작업 입력."""

    id: UUID
    patient_id: UUID
    storage_key: str
    content_hash: str
    size_bytes: int
    lease: UUID


class VoiceJobQueue(Protocol):
    """작업 선점과 늦은 응답의 조건부 저장을 정의한다."""

    def claim(self) -> VoiceJob | None:
        """처리할 작업 하나를 선점한다."""

    def complete(self, job: VoiceJob, draft: str | None, error: str | None) -> bool:
        """해당 실행이 여전히 유효할 때만 결과를 저장한다."""
