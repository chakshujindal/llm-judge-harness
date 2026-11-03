"""Provider tests run the real SDKs against an in-process mock HTTP transport (no network)."""

import json
from collections.abc import Callable
from pathlib import Path

import anthropic
import httpx2
import openai
import pytest

from ljh.errors import ConfigError, ProviderError, RetryableError
from ljh.llm.anthropic_client import AnthropicClient
from ljh.llm.base import LLMRequest, Message
from ljh.llm.openai_client import OpenAIClient

Handler = Callable[[httpx2.Request], httpx2.Response]
REQUEST = LLMRequest(
    system="Be brief.",
    messages=(Message(role="user", content="Capital of France?"),),
    max_tokens=256,
    sample=3,
)


def mock_http(handler: Handler, seen: list[httpx2.Request] | None = None) -> httpx2.AsyncClient:
    def record(request: httpx2.Request) -> httpx2.Response:
        if seen is not None:
            seen.append(request)
        return handler(request)

    return httpx2.AsyncClient(transport=httpx2.MockTransport(record))


def anthropic_client(handler: Handler, seen: list[httpx2.Request] | None = None) -> AnthropicClient:
    return AnthropicClient(
        "claude-sonnet-5-5", timeout_s=5, api_key="test-key", http_client=mock_http(handler, seen)
    )


def openai_client(handler: Handler, seen: list[httpx2.Request] | None = None) -> OpenAIClient:
    return OpenAIClient(
        "gpt-6.1-sol", timeout_s=5, api_key="test-key", http_client=mock_http(handler, seen)
    )


ANTHROPIC_OK = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": "Paris."},
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 21, "output_tokens": 4},
}

OPENAI_OK = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-6.1-sol",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Paris."}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 18, "completion_tokens": 3, "total_tokens": 21},
}


def error_body(kind: str) -> dict:
    return {"type": "error", "error": {"type": kind, "message": "nope"}}


async def test_anthropic_sends_request_and_reads_text_blocks() -> None:
    seen: list[httpx2.Request] = []
    client = anthropic_client(lambda _: httpx2.Response(200, json=ANTHROPIC_OK), seen)
    response = await client.complete(REQUEST)
    assert client.name == "anthropic:claude-sonnet-5-5"
    assert response.text == "Paris."  # the thinking block is skipped
    assert (response.input_tokens, response.output_tokens) == (21, 4)
    assert response.stop_reason == "end_turn"
    body = json.loads(seen[0].content)
    assert body == {
        "model": "claude-sonnet-5-5",
        "max_tokens": 256,
        "system": "Be brief.",
        "messages": [{"role": "user", "content": "Capital of France?"}],
    }
    assert len(seen) == 1


async def test_anthropic_omits_empty_system_and_keeps_refusals() -> None:
    seen: list[httpx2.Request] = []
    refusal = ANTHROPIC_OK | {"content": [], "stop_reason": "refusal"}
    client = anthropic_client(lambda _: httpx2.Response(200, json=refusal), seen)
    response = await client.complete(LLMRequest.single("hi"))
    assert response.text == ""
    assert response.stop_reason == "refusal"
    assert "system" not in json.loads(seen[0].content)


async def test_anthropic_429_is_retryable_with_retry_after() -> None:
    seen: list[httpx2.Request] = []
    client = anthropic_client(
        lambda _: httpx2.Response(
            429, headers={"retry-after": "7"}, json=error_body("rate_limit_error")
        ),
        seen,
    )
    with pytest.raises(RetryableError) as info:
        await client.complete(REQUEST)
    assert info.value.retry_after == 7.0
    assert len(seen) == 1  # the SDK's own retries are off


@pytest.mark.parametrize(("status", "fatal"), [(400, False), (401, True), (404, True)])
async def test_anthropic_client_errors_are_final(status: int, fatal: bool) -> None:
    client = anthropic_client(lambda _: httpx2.Response(status, json=error_body("x")))
    with pytest.raises(ProviderError) as info:
        await client.complete(REQUEST)
    assert not isinstance(info.value, RetryableError)
    assert info.value.fatal is fatal
    assert f"HTTP {status}" in str(info.value)


async def test_anthropic_connection_error_is_retryable() -> None:
    def boom(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    with pytest.raises(RetryableError, match="connection error"):
        await anthropic_client(boom).complete(REQUEST)


async def test_anthropic_other_sdk_errors_are_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    client = anthropic_client(lambda _: httpx2.Response(200, json=ANTHROPIC_OK))

    async def no_credentials(**_: object) -> None:
        raise anthropic.AnthropicError("no credentials found")

    monkeypatch.setattr(client._sdk.messages, "create", no_credentials)
    with pytest.raises(ProviderError, match="no credentials found") as info:
        await client.complete(REQUEST)
    assert info.value.fatal


async def test_openai_sends_request_and_reads_choice() -> None:
    seen: list[httpx2.Request] = []
    client = openai_client(lambda _: httpx2.Response(200, json=OPENAI_OK), seen)
    response = await client.complete(REQUEST)
    assert client.name == "openai:gpt-6.1-sol"
    assert response.text == "Paris."
    assert (response.input_tokens, response.output_tokens) == (18, 3)
    assert response.stop_reason == "stop"
    body = json.loads(seen[0].content)
    assert body["max_completion_tokens"] == 256
    assert body["messages"][0] == {"role": "system", "content": "Be brief."}
    assert "sample" not in body


async def test_openai_handles_missing_usage_and_content() -> None:
    reply = OPENAI_OK | {
        "usage": None,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None},
                "finish_reason": "length",
            }
        ],
    }
    response = await openai_client(lambda _: httpx2.Response(200, json=reply)).complete(
        LLMRequest.single("hi")
    )
    assert (response.text, response.input_tokens, response.output_tokens) == ("", 0, 0)


async def test_openai_reply_without_choices_is_an_error() -> None:
    empty = OPENAI_OK | {"choices": []}
    client = openai_client(lambda _: httpx2.Response(200, json=empty))
    with pytest.raises(ProviderError, match="returned no choices") as info:
        await client.complete(REQUEST)
    assert not info.value.fatal


async def test_openai_error_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    server_error = openai_client(
        lambda _: httpx2.Response(503, json={"error": {"message": "busy"}})
    )
    with pytest.raises(RetryableError, match="HTTP 503"):
        await server_error.complete(REQUEST)

    def boom(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("down", request=request)

    with pytest.raises(RetryableError, match="connection error"):
        await openai_client(boom).complete(REQUEST)

    other = openai_client(lambda _: httpx2.Response(200, json=OPENAI_OK))

    async def broken(**_: object) -> None:
        raise openai.OpenAIError("something odd")

    monkeypatch.setattr(other._sdk.chat.completions, "create", broken)
    with pytest.raises(ProviderError, match="something odd") as info:
        await other.complete(REQUEST)
    assert info.value.fatal


def test_openai_without_key_fails_early(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        OpenAIClient("gpt-6.1-sol", timeout_s=5)


def test_anthropic_without_credentials_fails_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(ConfigError, match="no Anthropic credentials found"):
        AnthropicClient("claude-sonnet-5-5", timeout_s=5)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))  # points at a missing profile
    with pytest.raises(ConfigError, match="cannot create the Anthropic client"):
        AnthropicClient("claude-sonnet-5-5", timeout_s=5)
