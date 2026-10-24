"""settings.database_url로 SQLAlchemy engine과 session factory를 만든다.

import할 때는 설정도 DB도 건드리지 않고, 함수를 처음 부를 때 만든다.
"""

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


@lru_cache
def get_engine() -> Engine:
    """engine을 한 번만 만들어 재사용한다. 실제 접속은 처음 쿼리할 때 일어난다."""
    # SQL 오류 메시지에 파라미터 값(문서 원문 등)이 붙지 않게 한다.
    return create_engine(get_settings().database_url, pool_pre_ping=True, hide_parameters=True)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    """session factory를 한 번만 만들어 재사용한다. commit은 session을 연 쪽이 한다."""
    return sessionmaker(get_engine())
