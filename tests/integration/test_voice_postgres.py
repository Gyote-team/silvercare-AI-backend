"""전용 PostgreSQL 스키마에서 lease·작업 선점·늦은 응답 차단을 검증한다."""

import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from app.infrastructure.db.repositories.voice_job_repository import VoiceJobRepository

URL = os.environ.get("VOICE_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="VOICE_TEST_DATABASE_URL is not configured")


@pytest.fixture
def queue():
    """테스트 전용 DB의 무작위 스키마만 만들고 정리한다."""
    if not make_url(URL).database.startswith("silvercare_voice_verify_"):
        pytest.fail("Use a dedicated silvercare_voice_verify_ database")
    schema = "voice_test_" + uuid4().hex
    admin = create_engine(URL, hide_parameters=True)
    with admin.begin() as db:
        db.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(URL, connect_args={"options": f"-csearch_path={schema}"},
                           hide_parameters=True)
    with engine.begin() as db:
        db.execute(text("""
            CREATE TABLE voice_transcriptions (
                id UUID PRIMARY KEY, patient_id UUID NOT NULL, storage_key TEXT NOT NULL,
                content_hash TEXT NOT NULL, size_bytes BIGINT NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'UPLOADED',
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                attempts INTEGER NOT NULL DEFAULT 0, lease_token UUID,
                error_code VARCHAR(50), draft_text TEXT, version BIGINT NOT NULL DEFAULT 0
            )
        """))
    try:
        yield VoiceJobRepository(engine), engine
    finally:
        engine.dispose()
        with admin.begin() as db:
            db.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def add(engine):
    """의료정보 없는 작업을 하나 추가한다."""
    identifier = uuid4()
    with engine.begin() as db:
        db.execute(text("""
            INSERT INTO voice_transcriptions(id,patient_id,storage_key,content_hash,size_bytes)
            VALUES (:id,:patient,'voice/test.wav',:hash,44)
        """), {"id": identifier, "patient": uuid4(), "hash": "0" * 64})
    return identifier


def test_workers_claim_distinct_jobs_and_result_is_single_use(queue):
    repo, engine = queue
    add(engine)
    add(engine)
    first, second = repo.claim(), repo.claim()
    assert first.id != second.id
    assert repo.claim() is None
    assert repo.complete(first, "초안", None)
    assert not repo.complete(first, "중복 결과", None)


def test_cancellation_discards_late_result(queue):
    repo, engine = queue
    add(engine)
    job = repo.claim()
    with engine.begin() as db:
        db.execute(text("UPDATE voice_transcriptions SET status='CANCELED' WHERE id=:id"),
                   {"id": job.id})
    assert not repo.complete(job, "늦은 결과", None)
    with engine.connect() as db:
        row = db.execute(text("SELECT status,draft_text FROM voice_transcriptions")).one()
        assert row.status == "CANCELED"
        assert row.draft_text is None


def test_expired_lease_can_recover_but_old_worker_cannot_save(queue):
    repo, engine = queue
    add(engine)
    old = repo.claim()
    with engine.begin() as db:
        db.execute(text("UPDATE voice_transcriptions SET updated_at=:past WHERE id=:id"),
                   {"past": datetime.now(UTC) - timedelta(minutes=11), "id": old.id})
    recovered = repo.claim()
    assert recovered.id == old.id
    assert recovered.lease != old.lease
    assert not repo.complete(old, "이전 실행", None)
    assert repo.complete(recovered, "복구된 실행", None)


def test_repeated_worker_crash_is_terminal_failure(queue):
    repo, engine = queue
    identifier = add(engine)
    with engine.begin() as db:
        db.execute(text("""
            UPDATE voice_transcriptions SET status='PROCESSING',attempts=3,updated_at=:past
            WHERE id=:id
        """), {"past": datetime.now(UTC) - timedelta(minutes=11), "id": identifier})
    assert repo.claim() is None
    with engine.connect() as db:
        row = db.execute(text("SELECT status,error_code FROM voice_transcriptions")).one()
        assert row.status == "FAILED"
        assert row.error_code == "WORKER_TIMEOUT"


def test_configured_lease_keeps_recent_job_and_recovers_only_after_expiry(queue):
    _, engine = queue
    repo = VoiceJobRepository(engine, lease_seconds=60)
    identifier = add(engine)
    original = repo.claim()
    with engine.begin() as db:
        db.execute(text("UPDATE voice_transcriptions SET updated_at=:past WHERE id=:id"),
                   {"past": datetime.now(UTC) - timedelta(seconds=30), "id": identifier})
    assert repo.claim() is None
    with engine.begin() as db:
        db.execute(text("UPDATE voice_transcriptions SET updated_at=:past WHERE id=:id"),
                   {"past": datetime.now(UTC) - timedelta(seconds=61), "id": identifier})
    recovered = repo.claim()
    assert recovered.id == original.id
    assert recovered.lease != original.lease
    assert not repo.complete(original, "expired result", None)
    assert repo.complete(recovered, "new result", None)
