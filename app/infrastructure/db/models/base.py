"""모든 ORM 모델이 상속하는 기반 클래스."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """ORM 모델 공통 기반. 테이블은 Spring Flyway가 만들므로 create_all을 부르지 않는다."""
