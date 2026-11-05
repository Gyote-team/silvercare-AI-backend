"""OpenAI 임베딩 어댑터 테스트. httpx.MockTransport로 대신해 실제 API를 부르지 않는다."""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from app.application.ports.text_embedder import (
    EMBEDDING_DIMENSIONS,
    EmbeddingError,
    EmbeddingErrorCode,
)
from app.core.config import Settings
from app.infrastructure.providers.embedding.openai_text_embedder import (
    OpenAITextEmbedder,
    create_text_embedder,
    has_embedding_api_key,
)

API_KEY = "sk-embedding-test-secret"
SOURCE_TEXT = "합성 검사 결과지 혈색소 13.2 g/dL"
MODEL = "text-embedding-3-small"

Handler = Callable[[httpx.Request], httpx.Response]


class Recorder:
    """받은 요청을 기록하고, 정해 둔 응답을 차례로 돌려주는 가짜 전송 계층."""

    def __init__(self, *responses: Handler) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """요청을 기록하고 다음 응답을 만든다. 남은 응답이 없으면 마지막 것을 되풀이한다."""
        self.requests.append(request)
        handler = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return handler(request)

    def bodies(self) -> list[dict[str, Any]]:
        """기록한 요청 본문을 JSON으로 읽는다."""
        return [json.loads(request.content) for request in self.requests]


def embeddings_ok(request: httpx.Request) -> httpx.Response:
    """입력마다 index를 값으로 채운 벡터를 거꾸로 된 순서로 돌려준다."""
    inputs = json.loads(request.content)["input"]
    data = [
        {"object": "embedding", "index": index, "embedding": [float(index)] * EMBEDDING_DIMENSIONS}
        for index in range(len(inputs))
    ]
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": list(reversed(data)),
            "model": MODEL,
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        },
    )


def status(code: int) -> Handler:
    """주어진 HTTP 상태와 원문을 담은 오류 응답을 만든다. 원문이 오류로 새지 않는지 본다."""
    return lambda request: httpx.Response(code, json={"error": {"message": SOURCE_TEXT}})


def timeout(request: httpx.Request) -> httpx.Response:
    """읽기 시간 초과를 낸다."""
    raise httpx.ReadTimeout("timed out", request=request)


def connection_error(request: httpx.Request) -> httpx.Response:
    """연결 실패를 낸다."""
    raise httpx.ConnectError("refused", request=request)


def make_embedder(
    recorder: Recorder,
    *,
    api_key: str | None = API_KEY,
    batch_size: int = 100,
    max_retries: int = 2,
    sleeps: list[float] | None = None,
) -> OpenAITextEmbedder:
    """가짜 전송 계층과 기록용 sleep을 넣은 어댑터를 만든다."""
    return OpenAITextEmbedder(
        api_key=SecretStr(api_key) if api_key is not None else None,
        model=MODEL,
        timeout_seconds=5,
        max_retries=max_retries,
        backoff_seconds=1.0,
        batch_size=batch_size,
        http_client=httpx.Client(transport=httpx.MockTransport(recorder)),
        sleep=(sleeps if sleeps is not None else []).append,
    )


def test_batches_keep_input_order() -> None:
    """batch_size씩 나눠 보내고, 응답 순서와 상관없이 입력 순서대로 벡터를 돌려준다."""
    recorder = Recorder(embeddings_ok)

    vectors = make_embedder(recorder, batch_size=2).embed(["a", "b", "c"])

    assert [body["input"] for body in recorder.bodies()] == [["a", "b"], ["c"]]
    assert [vector[0] for vector in vectors] == [0.0, 1.0, 0.0]
    assert all(len(vector) == EMBEDDING_DIMENSIONS for vector in vectors)


def test_request_uses_model_dimensions_and_embedding_key() -> None:
    """요청에 모델, 768 차원, EMBEDDING_API_KEY가 실린다."""
    recorder = Recorder(embeddings_ok)

    make_embedder(recorder).embed([SOURCE_TEXT])

    body = recorder.bodies()[0]
    assert (body["model"], body["dimensions"]) == (MODEL, EMBEDDING_DIMENSIONS)
    assert recorder.requests[0].headers["Authorization"] == f"Bearer {API_KEY}"


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (status(429), EmbeddingErrorCode.RATE_LIMITED),
        (status(500), EmbeddingErrorCode.API_ERROR),
        (status(503), EmbeddingErrorCode.API_ERROR),
        (timeout, EmbeddingErrorCode.TIMEOUT),
    ],
)
def test_transient_failure_is_retried_then_succeeds(failure: Handler, code: str) -> None:
    """429, 5xx, 시간 초과는 지수 대기 후 다시 불러 성공한다."""
    recorder = Recorder(failure, failure, embeddings_ok)
    sleeps: list[float] = []

    vectors = make_embedder(recorder, sleeps=sleeps).embed([SOURCE_TEXT])

    assert len(vectors) == 1
    assert len(recorder.requests) == 3
    assert sleeps == [1.0, 2.0]


def test_gives_up_after_max_retries() -> None:
    """재시도 횟수를 넘기면 마지막 오류 코드로 실패한다."""
    recorder = Recorder(status(429))

    with pytest.raises(EmbeddingError) as raised:
        make_embedder(recorder, max_retries=2).embed([SOURCE_TEXT])

    assert raised.value.code == EmbeddingErrorCode.RATE_LIMITED
    assert len(recorder.requests) == 3


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (status(400), EmbeddingErrorCode.REQUEST_REJECTED),
        (status(401), EmbeddingErrorCode.REQUEST_REJECTED),
        (connection_error, EmbeddingErrorCode.CONNECTION_FAILED),
    ],
)
def test_other_failures_are_not_retried(failure: Handler, code: EmbeddingErrorCode) -> None:
    """429 외 4xx와 연결 오류는 다시 부르지 않는다."""
    recorder = Recorder(failure)

    with pytest.raises(EmbeddingError) as raised:
        make_embedder(recorder).embed([SOURCE_TEXT])

    assert raised.value.code == code
    assert len(recorder.requests) == 1


def test_later_batch_failure_fails_whole_call() -> None:
    """뒤 묶음이 끝내 실패하면 앞 묶음이 성공했어도 전체가 실패한다."""
    recorder = Recorder(embeddings_ok, status(400))

    with pytest.raises(EmbeddingError):
        make_embedder(recorder, batch_size=1).embed(["a", "b"])


def test_wrong_dimension_is_rejected() -> None:
    """응답 벡터 차원이 768이 아니면 INVALID_RESPONSE다."""
    short = lambda request: httpx.Response(  # noqa: E731
        200, json={"object": "list", "data": [{"index": 0, "embedding": [0.1, 0.2]}]}
    )

    with pytest.raises(EmbeddingError) as raised:
        make_embedder(Recorder(short)).embed([SOURCE_TEXT])

    assert raised.value.code == EmbeddingErrorCode.INVALID_RESPONSE


def test_request_ignores_openai_base_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPENAI_BASE_URL을 다른 주소로 넣어도 요청은 api.openai.com으로 간다."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://other.example.test/v1")
    recorder = Recorder(embeddings_ok)

    make_embedder(recorder).embed([SOURCE_TEXT])

    assert str(recorder.requests[0].url) == "https://api.openai.com/v1/embeddings"


def test_duplicate_index_is_rejected() -> None:
    """index가 겹치면 개수가 맞아도 INVALID_RESPONSE다."""
    vector = [0.1] * EMBEDDING_DIMENSIONS
    duplicated = lambda request: httpx.Response(  # noqa: E731
        200,
        json={
            "object": "list",
            "data": [{"index": 0, "embedding": vector}, {"index": 0, "embedding": vector}],
        },
    )

    with pytest.raises(EmbeddingError) as raised:
        make_embedder(Recorder(duplicated)).embed(["a", "b"])

    assert raised.value.code == EmbeddingErrorCode.INVALID_RESPONSE


@pytest.mark.parametrize(("api_key", "expected"), [(None, False), ("  ", False), ("sk-x", True)])
def test_has_embedding_api_key_ignores_blank(
    test_settings: Settings, api_key: str | None, expected: bool
) -> None:
    """빈 값이나 공백 키는 키가 없는 것으로 본다."""
    settings = test_settings.model_copy(
        update={"embedding_api_key": SecretStr(api_key) if api_key is not None else None}
    )

    assert has_embedding_api_key(settings) is expected


def test_missing_vector_is_rejected() -> None:
    """입력보다 벡터가 적으면 INVALID_RESPONSE다."""
    empty = lambda request: httpx.Response(200, json={"object": "list", "data": []})  # noqa: E731

    with pytest.raises(EmbeddingError) as raised:
        make_embedder(Recorder(empty)).embed([SOURCE_TEXT])

    assert raised.value.code == EmbeddingErrorCode.INVALID_RESPONSE


@pytest.mark.parametrize("api_key", [None, "", "   ", "\t\n"])
def test_missing_key_sends_no_request(monkeypatch: pytest.MonkeyPatch, api_key: str | None) -> None:
    """키가 없거나 빈 값·공백이면 OPENAI_API_KEY가 있어도 쓰지 않고 MISSING_API_KEY로 실패한다."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-other-key")
    recorder = Recorder(embeddings_ok)
    embedder = make_embedder(recorder, api_key=api_key)

    with pytest.raises(EmbeddingError) as raised:
        embedder.embed([SOURCE_TEXT])

    assert raised.value.code == EmbeddingErrorCode.MISSING_API_KEY
    assert recorder.requests == []
    assert embedder._client is None


@pytest.mark.parametrize("env_value", ["", "   "])
def test_blank_key_in_env_becomes_missing_key(
    monkeypatch: pytest.MonkeyPatch, test_settings: Settings, env_value: str
) -> None:
    """환경 변수 EMBEDDING_API_KEY가 빈 값이나 공백이면 팩토리가 클라이언트를 만들지 않는다."""
    monkeypatch.setenv("EMBEDDING_API_KEY", env_value)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-other-key")
    settings = Settings(
        _env_file=None,
        database_url=test_settings.database_url,
        ai_internal_token=test_settings.ai_internal_token,
        storage_local_root=test_settings.storage_local_root,
    )

    embedder = create_text_embedder(settings)

    assert embedder._client is None
    with pytest.raises(EmbeddingError) as raised:
        embedder.embed([SOURCE_TEXT])
    assert raised.value.code == EmbeddingErrorCode.MISSING_API_KEY


def test_empty_input_sends_no_request() -> None:
    """입력이 비면 요청 없이 빈 목록을 돌려준다."""
    recorder = Recorder(embeddings_ok)

    assert make_embedder(recorder).embed([]) == []
    assert recorder.requests == []


def test_error_has_no_key_or_source_text() -> None:
    """오류 메시지와 원인에는 키와 원문이 없다."""
    with pytest.raises(EmbeddingError) as raised:
        make_embedder(Recorder(status(400))).embed([SOURCE_TEXT])

    assert str(raised.value) == EmbeddingErrorCode.REQUEST_REJECTED.value
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True


def test_factory_uses_only_embedding_settings(test_settings: Settings) -> None:
    """팩토리는 EMBEDDING_* 설정만 쓰고 다른 키는 쓰지 않는다."""
    settings = test_settings.model_copy(
        update={
            "openai_api_key": SecretStr("sk-openai"),
            "embedding_api_key": None,
            "embedding_model": "custom-embedding",
        }
    )

    embedder = create_text_embedder(settings)

    assert embedder.model_name == "custom-embedding"
    with pytest.raises(EmbeddingError) as raised:
        embedder.embed([SOURCE_TEXT])
    assert raised.value.code == EmbeddingErrorCode.MISSING_API_KEY
