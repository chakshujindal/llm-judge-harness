import json
from pathlib import Path

import pytest

from ljh.cache import CachedClient, ResponseCache, cache_key
from ljh.errors import ConfigError, ProviderError
from ljh.llm.base import LLMRequest
from ljh.llm.fake import (
    FALLBACK_ANSWER,
    FakeLLM,
    build_fake,
    echo,
    heuristic_judge,
    load_fake_answers,
)
from ljh.scoring.judge import build_judge_prompt
from ljh.suite import Rubric

EXAMPLES = Path(__file__).parent.parent / "examples"


async def test_echo_and_token_estimates() -> None:
    response = await FakeLLM("echo", echo).complete(LLMRequest.single("hello there", system="sys"))
    assert response.text == "hello there"
    assert response.input_tokens > 0
    assert response.output_tokens == len("hello there") // 4
    assert response.stop_reason == "end_turn"
    empty = await FakeLLM("empty", lambda _: "").complete(LLMRequest.single("x"))
    assert empty.output_tokens == 0


async def test_scripted_answers_match_on_prompt_text() -> None:
    model = build_fake("bot", {"bot": {"France": "Paris", "Japan": "Tokyo"}})
    assert model.name == "fake:bot"
    assert (await model.complete(LLMRequest.single("Capital of Japan?"))).text == "Tokyo"
    assert (await model.complete(LLMRequest.single("Capital of Peru?"))).text == FALLBACK_ANSWER


def test_build_fake_builtins_and_unknown() -> None:
    assert build_fake("echo", {}).name == "fake:echo"
    assert build_fake("judge", {}).name == "fake:judge"
    with pytest.raises(ConfigError, match="unknown fake model 'fake:nope'"):
        build_fake("nope", {"bot": {}})


def test_heuristic_judge_rewards_overlap_and_is_deterministic() -> None:
    rubric = Rubric.model_validate({"criteria": [{"name": "accuracy", "description": "d"}]})

    def judge(output: str, reference: str | None, sample: int = 0) -> dict:
        prompt = build_judge_prompt(rubric, "Q?", reference, output)
        return json.loads(heuristic_judge(LLMRequest.single(prompt, sample=sample)))

    good = judge("Tomás Ortega", "Tomás Ortega")["criteria"][0]
    bad = judge("Someone else entirely", "Tomás Ortega")["criteria"][0]
    assert good["name"] == "accuracy"
    assert good["score"] >= 4 > bad["score"]
    assert judge("x y", "x") == judge("x y", "x")
    no_reference = judge("an answer", None)["criteria"][0]
    assert "No reference" in no_reference["reasoning"]
    assert judge("", None)["criteria"][0]["score"] <= 2


def test_load_fake_answers(tmp_path: Path) -> None:
    answers = load_fake_answers(EXAMPLES / "fake" / "answers.yaml")
    assert {"strong", "weak"} <= set(answers)
    with pytest.raises(ConfigError, match="cannot read"):
        load_fake_answers(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("- a list\n")
    with pytest.raises(ConfigError, match="expected a mapping"):
        load_fake_answers(bad)


def test_cache_key_changes_with_model_prompt_and_params() -> None:
    base = LLMRequest.single("q", system="s", max_tokens=10)
    keys = {
        cache_key("m", base),
        cache_key("other", base),
        cache_key("m", LLMRequest.single("q2", system="s", max_tokens=10)),
        cache_key("m", LLMRequest.single("q", system="s", max_tokens=11)),
        cache_key("m", LLMRequest.single("q", system="s", max_tokens=10, sample=1)),
        cache_key("m", LLMRequest.single("q", system="other", max_tokens=10)),
    }
    assert len(keys) == 6
    assert cache_key("m", base) == cache_key("m", LLMRequest.single("q", system="s", max_tokens=10))


async def test_cached_client_serves_repeats_from_sqlite(tmp_path: Path) -> None:
    calls: list[str] = []

    def respond(request: LLMRequest) -> str:
        calls.append(request.messages[-1].content)
        return "answer"

    path = tmp_path / "cache.sqlite"
    with ResponseCache(path) as cache:
        client = CachedClient(FakeLLM("m", respond), cache)
        first = await client.complete(LLMRequest.single("q"))
        second = await client.complete(LLMRequest.single("q"))
    assert calls == ["q"]
    assert not first.cached
    assert second.cached
    assert second.text == "answer"
    with ResponseCache(path) as reopened:  # survives a restart
        assert reopened.get(cache_key("fake:m", LLMRequest.single("q"))) is not None


async def test_failed_calls_are_not_cached(tmp_path: Path) -> None:
    attempts: list[int] = []

    def flaky(request: LLMRequest) -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise ProviderError("HTTP 500")
        return "answer"

    with ResponseCache(tmp_path / "cache.sqlite") as cache:
        client = CachedClient(FakeLLM("m", flaky), cache)
        with pytest.raises(ProviderError):
            await client.complete(LLMRequest.single("q"))
        assert (await client.complete(LLMRequest.single("q"))).text == "answer"
    assert len(attempts) == 2


def test_cache_creates_its_folder_and_reports_bad_paths(tmp_path: Path) -> None:
    with ResponseCache(tmp_path / "new" / "dir" / "cache.sqlite"):
        assert (tmp_path / "new" / "dir" / "cache.sqlite").exists()
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(ConfigError, match="cannot open the response cache"):
        ResponseCache(blocker / "cache.sqlite")
