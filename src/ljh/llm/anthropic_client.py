"""Anthropic Messages API provider."""

from __future__ import annotations

import time
from typing import Any

import anthropic
import httpx2

from ljh.errors import ConfigError, ProviderError, RetryableError
from ljh.llm.base import LLMRequest, LLMResponse, error_for_status


class AnthropicClient:
    def __init__(
        self,
        model: str,
        *,
        timeout_s: float,
        api_key: str | None = None,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self.name = f"anthropic:{model}"
        self._model = model
        # max_retries=0: GuardedClient owns retries, so the SDK must not retry on its own too.
        # With no api_key the SDK resolves credentials itself (env var or `ant auth login`).
        try:
            self._sdk = anthropic.AsyncAnthropic(
                api_key=api_key, timeout=timeout_s, max_retries=0, http_client=http_client
            )
        except anthropic.AnthropicError as exc:  # e.g. a broken `ant` profile
            raise ConfigError(f"cannot create the Anthropic client: {exc}") from exc
        sdk = self._sdk
        if sdk.api_key is None and sdk.auth_token is None and sdk.credentials is None:
            raise ConfigError(
                "no Anthropic credentials found: set ANTHROPIC_API_KEY or run `ant auth login`"
            )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        # No temperature/top_p: current models only accept the defaults.
        params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if request.system:
            params["system"] = request.system
        started = time.perf_counter()
        try:
            message = await self._sdk.messages.create(**params)
        except anthropic.AnthropicError as exc:
            raise _to_provider_error(exc) from exc
        # Adaptive thinking may add thinking blocks; only text blocks are the answer.
        # A refusal comes back as stop_reason="refusal" with no text, and is scored as-is
        # (no fallback model, which would hide that this model declined).
        text = "".join(block.text for block in message.content if block.type == "text")
        return LLMResponse(
            text=text,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_s=time.perf_counter() - started,
            stop_reason=message.stop_reason,
        )


def _to_provider_error(exc: anthropic.AnthropicError) -> ProviderError:
    if isinstance(exc, anthropic.APIConnectionError):  # includes APITimeoutError
        return RetryableError(f"connection error: {exc}")
    if isinstance(exc, anthropic.APIStatusError):
        return error_for_status(exc.status_code, exc.message, exc.response.headers)
    # Anything else (e.g. no credentials found) would fail every case the same way.
    return ProviderError(f"{type(exc).__name__}: {exc}", fatal=True)
