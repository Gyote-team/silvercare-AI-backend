"""Spring Flyway가 생성한 작업 큐를 lease와 조건부 갱신으로 처리한다."""

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import Engine, text

from app.application.ports.voice_job_queue import VoiceJob


class VoiceJobRepository:
    """외부 API 호출 동안 DB 잠금을 유지하지 않는 영속 큐."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def claim(self) -> VoiceJob | None:
        """중복 작업자를 건너뛰고 재시작으로 남은 작업은 최대 세 번 복구한다."""
        now = datetime.now(UTC)
        cutoff = now - timedelta(minutes=10)
        with self.engine.begin() as db:
            db.execute(text("""
                UPDATE voice_transcriptions SET status='FAILED',error_code='WORKER_TIMEOUT',
                    lease_token=NULL,updated_at=:now,version=version+1
                WHERE status='PROCESSING' AND updated_at<:cutoff AND attempts>=3
            """), {"now": now, "cutoff": cutoff})
            row = db.execute(text("""
                SELECT id,patient_id,storage_key,content_hash,size_bytes
                FROM voice_transcriptions
                WHERE attempts<3 AND (status='UPLOADED' OR
                    (status='PROCESSING' AND updated_at<:cutoff))
                ORDER BY created_at,id LIMIT 1 FOR UPDATE SKIP LOCKED
            """), {"cutoff": cutoff}).mappings().first()
            if row is None:
                return None
            lease = uuid4()
            db.execute(text("""
                UPDATE voice_transcriptions SET status='PROCESSING',lease_token=:lease,
                    attempts=attempts+1,error_code=NULL,updated_at=:now,version=version+1
                WHERE id=:id
            """), {"lease": lease, "now": now, "id": row["id"]})
            return VoiceJob(
                UUID(str(row["id"])), UUID(str(row["patient_id"])), row["storage_key"],
                row["content_hash"], row["size_bytes"], lease,
            )

    def complete(self, job: VoiceJob, draft: str | None, error: str | None) -> bool:
        """취소·재시도 또는 다른 실행 뒤에 늦게 도착한 결과는 버린다."""
        with self.engine.begin() as db:
            result = db.execute(text("""
                UPDATE voice_transcriptions SET status=:status,draft_text=:draft,
                    error_code=:error,lease_token=NULL,updated_at=:now,version=version+1
                WHERE id=:id AND status='PROCESSING' AND lease_token=:lease
            """), {
                "status": "FAILED" if error else "DRAFT_READY", "draft": draft,
                "error": error, "now": datetime.now(UTC), "id": job.id, "lease": job.lease,
            })
            return result.rowcount == 1
