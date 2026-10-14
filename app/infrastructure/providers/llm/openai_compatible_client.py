"""Gemini 등 OpenAI 호환 Chat Completions API adapter."""

import json
from typing import Protocol

from openai import OpenAI

from app.core.config import Settings
from app.schemas.ai_chat import GeneratedAnswer


class LlmClient(Protocol):
    """provider 교체를 위한 최소 인터페이스다."""

    def generate_grounded_answer(self, system_prompt: str, user_prompt: str) -> GeneratedAnswer:
        """근거가 있는 구조화 답변을 생성한다."""


class OpenAiCompatibleLlmClient:
    """환경 변수로 받은 호환 endpoint에 요청한다."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    def generate_grounded_answer(self, system_prompt: str, user_prompt: str) -> GeneratedAnswer:
        response = self._client.chat.completions.create(
            model=self._settings.llm_model,
            temperature=self._settings.llm_temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM 응답 본문이 비어 있습니다.")
        return GeneratedAnswer.model_validate(json.loads(content))
