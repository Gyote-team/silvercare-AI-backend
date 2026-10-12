"""DB 통합 테스트 픽스처. 로컬 PostgreSQL에 붙고, 테스트마다 롤백해 데이터를 남기지 않는다."""

from collections.abc import Iterator
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.infrastructure.db.session import get_engine


@pytest.fixture
def db_session() -> Iterator[Session]:
    """롤백되는 트랜잭션 안의 session. 설정·드라이버·접속이 안 되면 skip한다."""
    try:
        connection = get_engine().connect()
    except ValidationError:
        pytest.skip("설정(.env)이 올바르지 않아 DB 테스트를 건너뜁니다.")
    except ImportError:
        pytest.skip("PostgreSQL 드라이버(psycopg)를 불러오지 못해 DB 테스트를 건너뜁니다.")
    except OperationalError:
        pytest.skip("DB에 접속할 수 없어 DB 테스트를 건너뜁니다.")
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def document_id(db_session: Session) -> UUID:
    """FK를 채우려고 users, patients, visits, documents 행을 하나씩 만든다."""
    user_id, patient_id, visit_id, document_id = uuid4(), uuid4(), uuid4(), uuid4()
    db_session.execute(
        # account_group_id는 NOT NULL이고 기본값이 없다. 자기 id를 넣는다.
        text(
            "INSERT INTO users (id, account_group_id, name, role)"
            " VALUES (:id, :id, 'test', 'PATIENT')"
        ),
        {"id": user_id},
    )
    db_session.execute(
        text("INSERT INTO patients (id, user_id, invite_code) VALUES (:id, :user_id, :code)"),
        {"id": patient_id, "user_id": user_id, "code": uuid4().hex[:16].upper()},
    )
    db_session.execute(
        text(
            "INSERT INTO visits (id, patient_id, visited_on)"
            " VALUES (:id, :patient_id, CURRENT_DATE)"
        ),
        {"id": visit_id, "patient_id": patient_id},
    )
    db_session.execute(
        text(
            "INSERT INTO documents (id, visit_id, uploader_user_id, patient_id,"
            " storage_key, file_name, mime_type, file_size_bytes)"
            " VALUES (:id, :visit_id, :user_id, :patient_id,"
            " :storage_key, 'original.pdf', 'application/pdf', 1)"
        ),
        {
            "id": document_id,
            "visit_id": visit_id,
            "user_id": user_id,
            "patient_id": patient_id,
            "storage_key": f"documents/{document_id}/original.pdf",
        },
    )
    return document_id


@pytest.fixture
def analysis_id(db_session: Session, document_id: UUID) -> UUID:
    """Spring이 업로드할 때 만드는 PENDING 분석 작업 행을 하나 만든다."""
    analysis_id = uuid4()
    db_session.execute(
        text(
            "INSERT INTO document_analyses (id, document_id, parser_version)"
            " VALUES (:id, :document_id, 'v1')"
        ),
        {"id": analysis_id, "document_id": document_id},
    )
    return analysis_id


@pytest.fixture
def session_factory(db_session: Session) -> sessionmaker[Session]:
    """스스로 commit하는 저장소에 넘길 session factory.

    테스트 연결에 묶여 있어 commit이 savepoint로만 끝나고, 테스트가 끝나면 함께 롤백된다.
    """
    return sessionmaker(bind=db_session.get_bind(), join_transaction_mode="create_savepoint")
