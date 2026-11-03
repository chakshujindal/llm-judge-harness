from pathlib import Path

import httpx2
import pytest

from ljh.cache import CachedClient, ResponseCache
from ljh.config import Settings
from ljh.errors import ConfigError
from ljh.llm.base import LLMRequest
from ljh.llm.factory import ClientFactory, parse_model_spec
from ljh.llm.fake import FakeLLM
from ljh.llm.resilience import GuardedClient
from ljh.pricing import DEFAULT_PRICES, Price, estimate_cost, load_prices

SETTINGS = Settings(anthropic_model="claude-sonnet-5-5", openai_model="gpt-6.1-sol")


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("anthropic", ("anthropic", "claude-sonnet-5-5")),
        ("Anthropic:claude-opus-5-5", ("anthropic", "claude-opus-5-5")),
        ("openai", ("openai", "gpt-6.1-sol")),
        ("openai:gpt-6.1-terra", ("openai", "gpt-6.1-terra")),
        ("fake:echo", ("fake", "echo")),
    ],
)
def test_parse_model_spec(spec: str, expected: tuple[str, str]) -> None:
    assert parse_model_spec(spec, SETTINGS) == expected


@pytest.mark.parametrize("spec", ["fake", "fake:", "gemini:pro", ""])
def test_parse_model_spec_rejects(spec: str) -> None:
    with pytest.raises(ConfigError, match="bad model spec"):
        parse_model_spec(spec, SETTINGS)


def test_factory_wraps_real_providers_but_not_fakes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with ResponseCache(tmp_path / "c.sqlite") as cache:
        factory = ClientFactory(SETTINGS, cache=cache)
        anthropic_model = factory.build("anthropic")
        openai_model = factory.build("openai")
        assert isinstance(anthropic_model, CachedClient)
        assert isinstance(openai_model, CachedClient)
        assert anthropic_model.name == "anthropic:claude-sonnet-5-5"
        assert isinstance(factory.build("fake:echo"), FakeLLM)
    uncached = ClientFactory(SETTINGS).build("anthropic:claude-opus-5-5")
    assert isinstance(uncached, GuardedClient)


def test_one_limiter_per_provider() -> None:
    factory = ClientFactory(SETTINGS)
    assert factory._limiter("anthropic") is factory._limiter("anthropic")
    assert factory._limiter("anthropic") is not factory._limiter("openai")


async def test_factory_client_end_to_end_with_retry_and_cache(tmp_path: Path) -> None:
    replies = iter(
        [
            httpx2.Response(
                429,
                headers={"retry-after": "0"},
                json={"type": "error", "error": {"type": "rate_limit_error", "message": "x"}},
            ),
            httpx2.Response(
                200,
                json={
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5-5",
                    "content": [{"type": "text", "text": "Paris"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 1},
                },
            ),
        ]
    )
    calls = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return next(replies)

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    with ResponseCache(tmp_path / "c.sqlite") as cache:
        factory = ClientFactory(
            Settings(max_retries=1, requests_per_minute=60_000), cache=cache, http_client=http
        )
        client = factory.build("anthropic")
        first = await client.complete(LLMRequest.single("Capital of France?"))
        second = await client.complete(LLMRequest.single("Capital of France?"))
    assert (first.text, first.cached, second.cached) == ("Paris", False, True)
    assert len(calls) == 2  # one 429, one success; the repeat came from the cache


def test_estimate_cost() -> None:
    prices = {"anthropic:m": Price(input=2.0, output=10.0)}
    assert estimate_cost("anthropic:m", 1_000_000, 100_000, prices) == pytest.approx(3.0)
    assert estimate_cost("fake:echo", 10, 10, prices) == 0.0
    assert estimate_cost("openai:unknown", 10, 10, prices) is None


def test_load_prices(tmp_path: Path) -> None:
    assert load_prices(None) == DEFAULT_PRICES
    file = tmp_path / "prices.yaml"
    file.write_text("openai:new-model: {input: 1.5, output: 6}\n")
    prices = load_prices(file)
    assert prices["openai:new-model"] == Price(1.5, 6.0)
    assert "anthropic:claude-sonnet-5-5" in prices
    file.write_text("openai:bad: {input: 1}\n")
    with pytest.raises(ConfigError, match="invalid prices file"):
        load_prices(file)
    with pytest.raises(ConfigError, match="invalid prices file"):
        load_prices(tmp_path / "missing.yaml")
