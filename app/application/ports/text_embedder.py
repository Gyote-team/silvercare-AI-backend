"""문장 목록을 벡터로 바꾸는 임베딩 포트와 예외를 정의한다."""

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

# Spring V2 embeddings.embedding vector(768)과 dimensions CHECK에 맞춘 값이다.
# 바꾸려면 Spring 마이그레이션이 먼저 필요하다.
EMBEDDING_DIMENSIONS = 768


class EmbeddingErrorCode(StrEnum):
    """임베딩이 실패한 이유."""

    MISSING_API_KEY = "MISSING_API_KEY"  # EMBEDDING_API_KEY가 설정되지 않음
    TIMEOUT = "TIMEOUT"  # 제한 시간 안에 응답이 없음
    RATE_LIMITED = "RATE_LIMITED"  # 호출 한도 초과(429)
    API_ERROR = "API_ERROR"  # provider 서버 오류(5xx)
    CONNECTION_FAILED = "CONNECTION_FAILED"  # 시간 초과가 아닌 네트워크 오류
    REQUEST_REJECTED = "REQUEST_REJECTED"  # 인증 실패, 잘못된 요청 등 429 외 4xx
    INVALID_RESPONSE = "INVALID_RESPONSE"  # 벡터 개수나 차원이 요청과 맞지 않음


class EmbeddingError(Exception):
    """임베딩 실패. 메시지에는 오류 코드만 담고 키와 원문은 담지 않는다."""

    def __init__(self, code: EmbeddingErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


class TextEmbedder(Protocol):
    """문장 목록을 EMBEDDING_DIMENSIONS 차원 벡터로 바꾸는 인터페이스.

    채팅 질문 검색도 같은 구현을 써야 문서 조각과 같은 공간의 벡터가 나온다.
    """

    @property
    def model_name(self) -> str:
        """embeddings.model_name에 기록할 모델 이름."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """입력 순서대로 벡터 목록을 반환한다. 실패하면 EmbeddingError를 던진다."""
        ...
