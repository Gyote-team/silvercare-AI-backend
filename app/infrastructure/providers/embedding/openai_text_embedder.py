"""OpenAI 임베딩 API로 문장을 768차원 벡터로 바꾼다. EMBEDDING_API_KEY만 쓴다."""

import time
from collections.abc import Callable, Sequence

import httpx
import openai
from openai import OpenAI
from pydantic import SecretStr

from app.application.ports.text_embedder import (
    EMBEDDING_DIMENSIONS,
    EmbeddingError,
    EmbeddingErrorCode,
)
from app.core.config import Settings

# 다시 부르면 성공할 수 있는 오류. 연결 오류와 그 밖의 4xx는 다시 보내도 결과가 같다고 본다.
_RETRYABLE_CODES = frozenset(
    {EmbeddingErrorCode.TIMEOUT, EmbeddingErrorCode.RATE_LIMITED, EmbeddingErrorCode.API_ERROR}
)


class OpenAITextEmbedder:
    """조각을 batch_size씩 묶어 요청하고, 재시도 대상 오류만 지수 대기 후 다시 부른다.

    키가 없으면 클라이언트를 만들지 않는다.
    SDK가 OPENAI_API_KEY 환경 변수로 대신하지 않게 하려는 것이다.
    """

    def __init__(
        self,
        *,
        api_key: SecretStr | None,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        backoff_seconds: float,
        batch_size: int,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        # .env의 EMBEDDING_API_KEY=처럼 빈 값이나 공백도 키가 없는 것으로 본다.
        # 빈 문자열을 SDK에 넘기면 OPENAI_API_KEY 환경 변수를 대신 읽으려 하기 때문이다.
        key = api_key.get_secret_value().strip() if api_key is not None else ""
        # SDK 재시도는 끄고 여기서 재시도 대상을 직접 고른다.
        self._client = (
            OpenAI(api_key=key, timeout=timeout_seconds, max_retries=0, http_client=http_client)
            if key
            else None
        )
        self._model = model
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._batch_size = batch_size
        self._sleep = sleep

    @property
    def model_name(self) -> str:
        """embeddings.model_name에 기록할 모델 이름."""
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """입력 순서대로 벡터 목록을 반환한다. 한 묶음이라도 실패하면 EmbeddingError를 던진다."""
        if not texts:
            return []
        if self._client is None:
            raise EmbeddingError(EmbeddingErrorCode.MISSING_API_KEY)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed_with_retry(texts[start : start + self._batch_size]))
        return vectors

    def close(self) -> None:
        """HTTP 연결을 닫는다."""
        if self._client is not None:
            self._client.close()

    def _embed_with_retry(self, batch: Sequence[str]) -> list[list[float]]:
        attempt = 0
        while True:
            try:
                return self._request(batch)
            except EmbeddingError as error:
                if error.code not in _RETRYABLE_CODES or attempt >= self._max_retries:
                    raise
            self._sleep(self._backoff_seconds * 2**attempt)
            attempt += 1

    def _request(self, batch: Sequence[str]) -> list[list[float]]:
        assert self._client is not None
        # SDK 예외 메시지에는 요청 정보가 섞일 수 있어 코드만 남기고 from None으로 끊는다.
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=list(batch),
                dimensions=EMBEDDING_DIMENSIONS,
                encoding_format="float",
            )
        except openai.APITimeoutError:
            raise EmbeddingError(EmbeddingErrorCode.TIMEOUT) from None
        except openai.APIConnectionError:
            raise EmbeddingError(EmbeddingErrorCode.CONNECTION_FAILED) from None
        except openai.RateLimitError:
            raise EmbeddingError(EmbeddingErrorCode.RATE_LIMITED) from None
        except openai.APIStatusError as error:
            code = (
                EmbeddingErrorCode.API_ERROR
                if error.status_code >= 500
                else EmbeddingErrorCode.REQUEST_REJECTED
            )
            raise EmbeddingError(code) from None
        return _validated_vectors(response, len(batch))


def _validated_vectors(response: object, expected_count: int) -> list[list[float]]:
    try:
        # 응답 순서는 보장되지 않으므로 index로 입력 순서를 맞춘다.
        data = sorted(response.data, key=lambda item: item.index)  # type: ignore[attr-defined]
        vectors = [list(item.embedding) for item in data]
    except (AttributeError, TypeError):
        raise EmbeddingError(EmbeddingErrorCode.INVALID_RESPONSE) from None
    if len(vectors) != expected_count or any(
        len(vector) != EMBEDDING_DIMENSIONS for vector in vectors
    ):
        raise EmbeddingError(EmbeddingErrorCode.INVALID_RESPONSE)
    return vectors


def create_text_embedder(settings: Settings) -> OpenAITextEmbedder:
    """EMBEDDING_* 설정만으로 임베딩 어댑터를 만든다."""
    return OpenAITextEmbedder(
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        timeout_seconds=settings.embedding_timeout_seconds,
        max_retries=settings.embedding_max_retries,
        backoff_seconds=settings.embedding_retry_backoff_seconds,
        batch_size=settings.embedding_batch_size,
    )
