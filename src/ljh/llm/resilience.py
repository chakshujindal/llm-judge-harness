"""Rate limiting, per-call timeouts and retries with backoff, wrapped around any LLMClient."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from ljh.errors import RetryableError
from ljh.llm.base import LLMClient, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 4
    base_delay_s: float = 1.0
    max_delay_s: float = 60.0  # also caps a server's Retry-After

    def backoff(self, attempt: int, rng: random.Random) -> float:
        """Exponential backoff with full jitter: uniform(0, min(cap, base * 2**attempt))."""
        ceiling = min(self.max_delay_s, self.base_delay_s * 2**attempt)
        return rng.uniform(0, ceiling)


class RateLimiter:
    """Caps in-flight calls and spaces call starts to stay under a requests-per-minute budget."""

    def __init__(
        self,
        max_concurrency: int,
        requests_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._interval_s = 60.0 / requests_per_minute
        self._next_start = 0.0
        self._lock = asyncio.Lock()
        self._clock = clock
        self._sleep = sleep

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        async with self._semaphore:
            await self._wait_for_turn()
            yield

    async def _wait_for_turn(self) -> None:
        async with self._lock:
            now = self._clock()
            start = max(now, self._next_start)
            self._next_start = start + self._interval_s
        if start > now:
            await self._sleep(start - now)


class GuardedClient:
    """Wraps a provider client with a rate limiter, a per-call timeout and retries."""

    def __init__(
        self,
        inner: LLMClient,
        limiter: RateLimiter,
        policy: RetryPolicy,
        timeout_s: float,
        *,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.name = inner.name
        self._inner = inner
        self._limiter = limiter
        self._policy = policy
        self._timeout_s = timeout_s
        self._sleep = sleep
        self._rng = rng or random.Random()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        attempt = 0
        while True:
            try:
                return await self._attempt(request)
            except RetryableError as exc:
                if attempt >= self._policy.max_retries:
                    raise
                if exc.retry_after is not None:
                    delay = min(exc.retry_after, self._policy.max_delay_s)
                else:
                    delay = self._policy.backoff(attempt, self._rng)
                logger.info(
                    "%s: %s; retry %d/%d in %.1fs",
                    self.name,
                    exc,
                    attempt + 1,
                    self._policy.max_retries,
                    delay,
                )
            await self._sleep(delay)
            attempt += 1

    async def _attempt(self, request: LLMRequest) -> LLMResponse:
        async with self._limiter.slot():
            # The SDKs get the same timeout for their HTTP calls; this one also covers
            # anything outside them (e.g. a fake model, or a slow connection pool).
            try:
                async with asyncio.timeout(self._timeout_s):
                    return await self._inner.complete(request)
            except TimeoutError as exc:
                raise RetryableError(f"no response within {self._timeout_s:g}s") from exc
