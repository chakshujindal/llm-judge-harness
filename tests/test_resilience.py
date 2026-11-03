import asyncio
import logging
import random

import pytest

from ljh.errors import ProviderError, RetryableError
from ljh.llm.base import LLMRequest, LLMResponse, error_for_status, parse_retry_after
from ljh.llm.resilience import GuardedClient, RateLimiter, RetryPolicy

REQUEST = LLMRequest.single("q")


class Scripted:
    """Raises the queued errors in order, then answers."""

    name = "test:model"

    def __init__(self, *errors: Exception, delay_s: float = 0.0) -> None:
        self.errors = list(errors)
        self.calls = 0
        self.delay_s = delay_s

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.errors:
            raise self.errors.pop(0)
        return LLMResponse(text="ok")


class Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def guarded(
    inner: Scripted, sleeps: Sleeps, retries: int = 3, timeout_s: float = 5.0
) -> GuardedClient:
    limiter = RateLimiter(max_concurrency=4, requests_per_minute=60_000, sleep=Sleeps())
    return GuardedClient(
        inner,
        limiter,
        RetryPolicy(max_retries=retries, base_delay_s=1.0, max_delay_s=8.0),
        timeout_s,
        sleep=sleeps,
        rng=random.Random(0),
    )


class UpperBound(random.Random):
    """An rng that always picks the top of the jitter range, to expose the ceiling."""

    def uniform(self, a: float, b: float) -> float:
        return b


def test_backoff_doubles_up_to_the_cap() -> None:
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=4.0)
    assert [policy.backoff(attempt, UpperBound()) for attempt in range(5)] == [1, 2, 4, 4, 4]


def test_backoff_is_jittered_below_the_ceiling() -> None:
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=4.0)
    rng = random.Random(1)
    delays = [policy.backoff(3, rng) for _ in range(50)]
    assert all(0 <= d <= 4.0 for d in delays)
    assert len(set(delays)) > 1


async def test_retries_retryable_errors_then_succeeds(caplog: pytest.LogCaptureFixture) -> None:
    inner = Scripted(RetryableError("HTTP 529"), RetryableError("HTTP 429", retry_after=2.5))
    sleeps = Sleeps()
    with caplog.at_level(logging.INFO):
        response = await guarded(inner, sleeps).complete(REQUEST)
    assert response.text == "ok"
    assert inner.calls == 3
    assert len(sleeps.calls) == 2
    assert 0 <= sleeps.calls[0] <= 1.0  # jittered backoff for attempt 0
    assert sleeps.calls[1] == 2.5  # Retry-After wins over computed backoff
    assert "retry 1/3" in caplog.text


async def test_retry_after_is_capped() -> None:
    inner = Scripted(RetryableError("HTTP 429", retry_after=3600))
    sleeps = Sleeps()
    await guarded(inner, sleeps).complete(REQUEST)
    assert sleeps.calls == [8.0]  # the policy's max_delay_s, not an hour


async def test_gives_up_after_max_retries() -> None:
    inner = Scripted(*(RetryableError(f"HTTP 500 #{i}") for i in range(5)))
    with pytest.raises(RetryableError, match="#2"):
        await guarded(inner, Sleeps(), retries=2).complete(REQUEST)
    assert inner.calls == 3


async def test_non_retryable_errors_are_not_retried() -> None:
    inner = Scripted(ProviderError("HTTP 400: bad"))
    with pytest.raises(ProviderError, match="HTTP 400"):
        await guarded(inner, Sleeps()).complete(REQUEST)
    assert inner.calls == 1


async def test_timeout_becomes_retryable() -> None:
    inner = Scripted(delay_s=0.05)
    with pytest.raises(RetryableError, match=r"no response within 0\.01s"):
        await guarded(inner, Sleeps(), retries=1, timeout_s=0.01).complete(REQUEST)
    assert inner.calls == 2


async def test_rate_limiter_spaces_call_starts() -> None:
    now = [100.0]
    sleeps = Sleeps()
    limiter = RateLimiter(
        max_concurrency=10, requests_per_minute=60, clock=lambda: now[0], sleep=sleeps
    )
    for _ in range(3):
        async with limiter.slot():
            pass
    assert sleeps.calls == [1.0, 2.0]  # starts at t=100, 101, 102


async def test_rate_limiter_caps_concurrency() -> None:
    limiter = RateLimiter(max_concurrency=2, requests_per_minute=60_000)
    active = peak = 0

    async def work() -> None:
        nonlocal active, peak
        async with limiter.slot():
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(*(work() for _ in range(6)))
    assert peak == 2


@pytest.mark.parametrize(
    ("status", "retryable", "fatal"),
    [
        (429, True, False),
        (529, True, False),
        (408, True, False),
        (409, False, False),
        (400, False, False),
        (401, False, True),
        (404, False, True),
    ],
)
def test_error_for_status(status: int, retryable: bool, fatal: bool) -> None:
    error = error_for_status(status, "msg", {"retry-after": "3"})
    assert isinstance(error, RetryableError) is retryable
    assert error.fatal is fatal
    if retryable:
        assert isinstance(error, RetryableError)
        assert error.retry_after == 3.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("2", 2.0), ("0.5", 0.5), ("-1", None), ("Wed, 21 Oct 2026 07:28:00 GMT", None)],
)
def test_parse_retry_after(value: str | None, expected: float | None) -> None:
    assert parse_retry_after(value) == expected
