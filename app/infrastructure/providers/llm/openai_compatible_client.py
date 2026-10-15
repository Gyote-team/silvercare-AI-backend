"""Gemini 등 OpenAI 호환 Chat Completions API adapter."""

import json
from typing import Protocol

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from app.contracts.ai_result_document import GeneratedAnswer
from app.core.config import Settings
from app.domain.ai_result.error import AiResultErrorCode, AiResultGenerationError


class LlmClient(Protocol):
    """provider 교체를 위한 최소 인터페이스다."""

    def generate_grounded_answer(self, system_prompt: str, user_prompt: str) -> GeneratedAnswer:
        """근거가 있는 구조화 답변을 생성한다."""


class OpenAiCompatibleLlmClient:
    """환경 변수로 받은 호환 endpoint에 요청한다."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else "",
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    def generate_grounded_answer(self, system_prompt: str, user_prompt: str) -> GeneratedAnswer:
        if self._settings.llm_api_key is None:
            raise AiResultGenerationError(
                AiResultErrorCode.AI_API_KEY_MISSING, retryable=False
            )
        try:
            request_kwargs = {
                "model": self._settings.llm_model,
                "temperature": self._settings.llm_temperature,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            # Gemini OpenAI 호환 API는 OpenAI 전용 reasoning_effort를 받지 않는다.
            if self._settings.llm_provider != "gemini_openai_compatible":
                request_kwargs["reasoning_effort"] = self._settings.llm_reasoning_effort
            response = self._client.chat.completions.create(**request_kwargs)
        except APITimeoutError:
            raise AiResultGenerationError(AiResultErrorCode.AI_TIMEOUT, retryable=True) from None
        except RateLimitError:
            raise AiResultGenerationError(AiResultErrorCode.AI_RATE_LIMIT, retryable=True) from None
        except AuthenticationError:
            raise AiResultGenerationError(
                AiResultErrorCode.AI_AUTH_FAILED, retryable=False
            ) from None
        except BadRequestError:
            raise AiResultGenerationError(
                AiResultErrorCode.AI_REQUEST_REJECTED, retryable=False
            ) from None
        except APIConnectionError:
            raise AiResultGenerationError(
                AiResultErrorCode.AI_CONNECTION_FAILED, retryable=True
            ) from None
        except APIStatusError:
            raise AiResultGenerationError(
                AiResultErrorCode.AI_GENERATION_FAILED, retryable=True
            ) from None
        content = response.choices[0].message.content
        if not content:
            raise AiResultGenerationError(
                AiResultErrorCode.AI_RESPONSE_INVALID, retryable=True
            )
        try:
            return GeneratedAnswer.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValueError):
            raise AiResultGenerationError(
                AiResultErrorCode.AI_RESPONSE_INVALID, retryable=True
            ) from None
