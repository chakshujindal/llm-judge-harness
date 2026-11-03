"""OpenAI Chat Completions provider."""

from __future__ import annotations

import time
from typing import Any

import httpx2
import openai

from ljh.errors import ConfigError, ProviderError, RetryableError
from ljh.llm.base import LLMRequest, LLMResponse, error_for_status


class OpenAIClient:
    def __init__(
        self,
        model: str,
        *,
        timeout_s: float,
        api_key: str | None = None,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self.name = f"openai:{model}"
        self._model = model
        try:
            # max_retries=0: GuardedClient owns retries.
            self._sdk = openai.AsyncOpenAI(
                api_key=api_key, timeout=timeout_s, max_retries=0, http_client=http_client
            )
        except openai.OpenAIError as exc:
            raise ConfigError(
                f"cannot create the OpenAI client (set OPENAI_API_KEY): {exc}"
            ) from exc

    async def complete(self, request: LLMRequest) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend({"role": m.role, "content": m.content} for m in request.messages)
        params: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            # Reasoning models count hidden reasoning here too, so keep it generous.
            "max_completion_tokens": request.max_tokens,
        }
        started = time.perf_counter()
        try:
            completion = await self._sdk.chat.completions.create(**params)
        except openai.OpenAIError as exc:
            raise _to_provider_error(exc) from exc
        if not completion.choices:
            raise ProviderError(f"{self.name} returned no choices")
        choice = completion.choices[0]
        usage = completion.usage
        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_s=time.perf_counter() - started,
            stop_reason=choice.finish_reason,
        )


def _to_provider_error(exc: openai.OpenAIError) -> ProviderError:
    if isinstance(exc, openai.APIConnectionError):  # includes APITimeoutError
        return RetryableError(f"connection error: {exc}")
    if isinstance(exc, openai.APIStatusError):
        return error_for_status(exc.status_code, exc.message, exc.response.headers)
    return ProviderError(f"{type(exc).__name__}: {exc}", fatal=True)
