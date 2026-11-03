"""Builds LLM clients from "provider:model" specs such as anthropic:claude-sonnet-5-5."""

from __future__ import annotations

from collections.abc import Mapping

import httpx2

from ljh.cache import CachedClient, ResponseCache
from ljh.config import Settings
from ljh.errors import ConfigError
from ljh.llm.anthropic_client import AnthropicClient
from ljh.llm.base import LLMClient
from ljh.llm.fake import build_fake
from ljh.llm.openai_client import OpenAIClient
from ljh.llm.resilience import GuardedClient, RateLimiter, RetryPolicy


def parse_model_spec(spec: str, settings: Settings) -> tuple[str, str]:
    """Split "provider:model". A bare "anthropic" or "openai" uses the env-configured default."""
    provider, _, model = spec.strip().partition(":")
    provider = provider.lower()
    if provider == "anthropic":
        return provider, model or settings.anthropic_model
    if provider == "openai":
        return provider, model or settings.openai_model
    if provider == "fake" and model:
        return provider, model
    raise ConfigError(
        f"bad model spec {spec!r}: use anthropic[:model], openai[:model] or fake:<name>"
    )


class ClientFactory:
    """Real providers share one rate limiter per provider plus the response cache.

    Fake models skip both: they are free, instant and should not pollute the cache.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        cache: ResponseCache | None = None,
        fake_answers: Mapping[str, Mapping[str, str]] | None = None,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._cache = cache
        self._fake_answers = fake_answers or {}
        self._http_client = http_client  # tests inject a mock transport here
        self._limiters: dict[str, RateLimiter] = {}

    def build(self, spec: str) -> LLMClient:
        provider, model = parse_model_spec(spec, self._settings)
        if provider == "fake":
            return build_fake(model, self._fake_answers)
        settings = self._settings
        inner: LLMClient
        if provider == "anthropic":
            inner = AnthropicClient(
                model, timeout_s=settings.timeout_s, http_client=self._http_client
            )
        else:
            inner = OpenAIClient(model, timeout_s=settings.timeout_s, http_client=self._http_client)
        guarded = GuardedClient(
            inner,
            self._limiter(provider),
            RetryPolicy(max_retries=settings.max_retries),
            settings.timeout_s,
        )
        return CachedClient(guarded, self._cache) if self._cache else guarded

    def _limiter(self, provider: str) -> RateLimiter:
        if provider not in self._limiters:
            self._limiters[provider] = RateLimiter(
                self._settings.max_concurrency, self._settings.requests_per_minute
            )
        return self._limiters[provider]
