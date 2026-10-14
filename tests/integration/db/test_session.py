"""DB engine 설정 테스트. 로컬 PostgreSQL에 붙고, 값은 모두 합성 데이터다."""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session

PARAMETER_VALUE = "합성 원문 혈색소 13.5"


def test_sql_error_message_has_no_parameter_values(db_session: Session) -> None:
    """SQL이 실패해도 예외 메시지에 파라미터 값이 들어가지 않는다."""
    with pytest.raises(ProgrammingError) as error:
        db_session.execute(
            text("INSERT INTO table_that_does_not_exist (ocr_text) VALUES (:ocr_text)"),
            {"ocr_text": PARAMETER_VALUE},
        )

    assert PARAMETER_VALUE not in str(error.value)
