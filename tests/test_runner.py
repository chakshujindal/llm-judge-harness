import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import pytest
from helpers import make_suite

from ljh.errors import ConfigError, ProviderError, RunAbortedError
from ljh.llm.base import LLMRequest, LLMResponse
from ljh.llm.fake import FakeLLM, build_fake, heuristic_judge
from ljh.pricing import DEFAULT_PRICES, Price
from ljh.results import CaseResult
from ljh.runner import run_suite
from ljh.scoring import JudgeContext

ANSWERS = {"good": {"France": "Paris", "Japan": "Tokyo"}, "half": {"France": "Paris"}}
NOW = datetime(2026, 11, 2, 9, 30, tzinfo=UTC)


async def run(suite: Any, models: list[Any], **kwargs: Any) -> Any:
    defaults: dict[str, Any] = {
        "judge": None,
        "prices": DEFAULT_PRICES,
        "max_tokens": 100,
        "concurrency": 2,
        "now": NOW,
    }
    defaults.update(kwargs)
    return await run_suite(suite, models, **defaults)


async def test_runs_every_model_on_every_case_in_order() -> None:
    progress: list[tuple[str, int, int]] = []

    def on_result(result: CaseResult, done: int, total: int) -> None:
        progress.append((result.case_id, done, total))

    models = [build_fake("good", ANSWERS), build_fake("half", ANSWERS)]
    result = await run(make_suite(), models, on_result=on_result)
    assert result.run_id == "20261102T093000Z"
    assert result.models == ["fake:good", "fake:half"]
    assert [(r.model, r.case_id, r.passed) for r in result.results] == [
        ("fake:good", "france", True),
        ("fake:good", "japan", True),
        ("fake:half", "france", True),
        ("fake:half", "japan", False),
    ]
    assert sorted(done for _, done, _ in progress) == [1, 2, 3, 4]
    assert {total for _, _, total in progress} == {4}
    first = result.results[0]
    assert first.cost_usd == 0.0
    assert first.input_tokens > 0
    assert first.stop_reason == "end_turn"
    assert result.judge_model is None


async def test_suite_max_tokens_and_system_prompt_reach_the_model() -> None:
    seen: list[LLMRequest] = []

    def respond(request: LLMRequest) -> str:
        seen.append(request)
        return "Paris"

    await run(make_suite(max_tokens=42), [FakeLLM("spy", respond)])
    assert {r.max_tokens for r in seen} == {42}
    assert {r.system for r in seen} == {"Be brief."}


async def test_priced_model_gets_a_cost_estimate() -> None:
    class Priced:
        name = "anthropic:test"

        async def complete(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(text="Paris", input_tokens=1_000_000, output_tokens=0)

    prices = {"anthropic:test": Price(2.0, 10.0)}
    result = await run(make_suite(), [Priced()], prices=prices)
    assert result.results[0].cost_usd == pytest.approx(2.0)


class Failing:
    name = "anthropic:broken"

    def __init__(self, error: ProviderError) -> None:
        self.error = error

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise self.error


async def test_non_fatal_error_is_recorded_per_case(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR):
        result = await run(make_suite(), [Failing(ProviderError("HTTP 400: too long"))])
    assert all(r.error == "HTTP 400: too long" and not r.passed for r in result.results)
    assert "model=anthropic:broken case=france" in caplog.text


async def test_fatal_error_aborts_the_run() -> None:
    with pytest.raises(RunAbortedError, match="anthropic:broken on case 'france': HTTP 401"):
        await run(make_suite(), [Failing(ProviderError("HTTP 401: bad key", fatal=True))])


async def test_judge_is_required_when_the_suite_uses_one() -> None:
    suite = make_suite(
        default_scorers=[
            {
                "type": "llm_judge",
                "rubric": {"criteria": [{"name": "accuracy", "description": "Right?"}]},
            }
        ]
    )
    with pytest.raises(ConfigError, match="judge model is required"):
        await run(suite, [build_fake("good", ANSWERS)])
    judge = JudgeContext(FakeLLM("judge", heuristic_judge), DEFAULT_PRICES, 100)
    result = await run(suite, [build_fake("good", ANSWERS)], judge=judge)
    assert result.judge_model == "fake:judge"
    assert all(r.scores[0].judge is not None for r in result.results)


class SlowFirstCase:
    """The first case answers slowest, so completion order differs from suite order."""

    name = "fake:slow"

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.05 if "case a" in request.messages[0].content else 0.0)
        self.active -= 1
        return LLMResponse(text="ok")


async def test_results_keep_suite_order_and_concurrency_is_bounded() -> None:
    ids = ["a", "b", "c", "d"]
    suite = make_suite(cases=[{"id": i, "input": f"case {i}", "reference": "ok"} for i in ids])
    model = SlowFirstCase()
    finished: list[str] = []
    result = await run(
        suite, [model], concurrency=2, on_result=lambda r, *_: finished.append(r.case_id)
    )
    assert finished[-1] == "a"  # the slow case finished last...
    assert [r.case_id for r in result.results] == ids  # ...but results keep suite order
    assert model.peak == 2
