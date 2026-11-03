"""The provider-neutral LLM interface used everywhere in the harness."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ljh.errors import ProviderError, RetryableError

FATAL_STATUSES = frozenset({401, 403, 404})


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    """Everything that decides a model's answer. Its JSON form is the cache key."""

    model_config = ConfigDict(frozen=True)

    system: str | None = None
    messages: tuple[Message, ...]
    max_tokens: int = Field(default=1024, gt=0)
    # Repeat index for judge sampling: part of the cache key, never sent to the provider.
    sample: int = Field(default=0, ge=0)

    @classmethod
    def single(
        cls, prompt: str, *, system: str | None = None, max_tokens: int = 1024, sample: int = 0
    ) -> LLMRequest:
        """A one-turn request with a single user message."""
        return cls(
            system=system,
            messages=(Message(role="user", content=prompt),),
            max_tokens=max_tokens,
            sample=sample,
        )


class LLMResponse(BaseModel):
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    stop_reason: str | None = None
    cached: bool = False


class LLMClient(Protocol):
    """One model behind one provider. `name` is the "provider:model" spec."""

    name: str

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


def error_for_status(status: int, message: str, headers: Mapping[str, str]) -> ProviderError:
    """Map an HTTP error status to a retryable or a final ProviderError."""
    if status in (408, 429) or status >= 500:
        return RetryableError(
            f"HTTP {status}: {message}", retry_after=parse_retry_after(headers.get("retry-after"))
        )
    return ProviderError(f"HTTP {status}: {message}", fatal=status in FATAL_STATUSES)


def parse_retry_after(value: str | None) -> float | None:
    """Read a Retry-After header given in seconds; other forms are ignored."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None
