import json
import logging
from typing import Any

import pytest

from ljh.errors import ConfigError, ProviderError
from ljh.llm.base import LLMRequest
from ljh.llm.fake import FakeLLM, heuristic_judge
from ljh.pricing import DEFAULT_PRICES
from ljh.results import CriterionVerdict, JudgeRun
from ljh.scoring import apply_scorer
from ljh.scoring.judge import (
    JudgeContext,
    aggregate,
    build_judge_prompt,
    judge_case,
    parse_verdict,
    weighted_score,
)
from ljh.suite import Case, JudgeSpec, Rubric

RUBRIC = Rubric.model_validate(
    {
        "criteria": [
            {"name": "accuracy", "description": "Right facts.", "weight": 3},
            {"name": "clarity", "description": "Easy to read.", "weight": 1},
        ],
        "pass_threshold": 0.7,
    }
)
CASE = Case(id="c1", input="Capital of France?", reference="Paris")


def verdict(accuracy: int, clarity: int, reasoning: str = "ok") -> str:
    return json.dumps(
        {
            "criteria": [
                {"name": "accuracy", "score": accuracy, "reasoning": reasoning},
                {"name": "clarity", "score": clarity, "reasoning": reasoning},
            ]
        }
    )


def judge_spec(**overrides: Any) -> JudgeSpec:
    data: dict[str, Any] = {"type": "llm_judge", "rubric": RUBRIC.model_dump()}
    data.update(overrides)
    return JudgeSpec.model_validate(data)


def scripted_judge(replies: list[str]) -> tuple[FakeLLM, list[LLMRequest]]:
    """A fake judge that returns `replies` in order and records the requests it saw."""
    seen: list[LLMRequest] = []

    def respond(request: LLMRequest) -> str:
        seen.append(request)
        return replies[len(seen) - 1]

    return FakeLLM("scripted-judge", respond), seen


def context(client: Any) -> JudgeContext:
    return JudgeContext(client=client, prices=DEFAULT_PRICES, max_tokens=512)


def test_prompt_contains_rubric_reference_and_response() -> None:
    prompt = build_judge_prompt(RUBRIC, "Q?", "Paris", "It is Paris.")
    assert '<criterion name="accuracy" weight="3">Right facts.</criterion>' in prompt
    assert "<reference>\nParis\n</reference>" in prompt
    assert prompt.endswith("<response>\nIt is Paris.\n</response>")
    no_ref = build_judge_prompt(RUBRIC, "Q?", None, "x")
    assert "<reference>" not in no_ref
    assert "no reference answer" in no_ref


def test_parse_verdict_accepts_surrounding_prose() -> None:
    criteria = parse_verdict("Here you go:\n" + verdict(5, 4) + "\nThanks", RUBRIC)
    assert [c.score for c in criteria] == [5, 4]


@pytest.mark.parametrize(
    "text",
    [
        "no json here",
        '{"criteria": [{"name": "accuracy", "score": 9, "reasoning": "x"}]}',
        '{"criteria": [{"name": "accuracy", "score": 5, "reasoning": "x"}]}',
        verdict(5, 5, reasoning="x" * 501),
    ],
)
def test_parse_verdict_rejects_bad_output(text: str) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - pydantic and our own checks both raise ValueError
        parse_verdict(text, RUBRIC)


def test_weighted_score_normalises_to_unit_range() -> None:
    def criteria(a: int, c: int) -> list[CriterionVerdict]:
        return [
            CriterionVerdict(name="accuracy", score=a, reasoning=""),
            CriterionVerdict(name="clarity", score=c, reasoning=""),
        ]

    assert weighted_score(criteria(5, 5), RUBRIC) == 1.0
    assert weighted_score(criteria(1, 1), RUBRIC) == 0.0
    assert weighted_score(criteria(5, 1), RUBRIC) == pytest.approx(0.75)


def run(score: float, threshold: float = 0.7) -> JudgeRun:
    return JudgeRun(score=score, passed=score >= threshold, criteria=[])


def test_aggregate_mean_vs_majority() -> None:
    runs = [run(1.0), run(0.6), run(0.6)]  # mean 0.733, but only 1 of 3 runs passed
    assert aggregate(runs, "mean", 0.7) == (
        True,
        pytest.approx(0.7333, abs=1e-3),
        pytest.approx(2 / 3),
    )
    passed, _, agreement = aggregate(runs, "majority", 0.7)
    assert not passed
    assert agreement == pytest.approx(2 / 3)
    assert aggregate([run(0.8), run(0.9)], "majority", 0.7)[0]
    assert not aggregate([run(0.8), run(0.1)], "majority", 0.7)[0]  # a tie is not a majority


async def test_judge_case_single_run() -> None:
    client, seen = scripted_judge([verdict(5, 3)])
    result = await judge_case(judge_spec(), CASE, "Paris", context(client))
    assert result.passed
    assert result.score == pytest.approx(0.875)
    assert result.judge is not None
    assert result.judge.agreement == 1.0
    assert result.judge.calls == 1
    assert result.judge.cost_usd == 0.0  # fake models are free
    assert result.judge.criterion_means == {"accuracy": 5.0, "clarity": 3.0}
    assert result.detail == "mean 0.88 vs threshold 0.7 over 1 run(s); agreement 100%"
    assert seen[0].system is not None
    assert seen[0].max_tokens == 512


async def test_judge_repeats_use_distinct_samples_and_report_agreement() -> None:
    client, seen = scripted_judge([verdict(5, 5), verdict(2, 2), verdict(5, 4)])
    result = await judge_case(
        judge_spec(repeats=3, aggregate="majority"), CASE, "Paris", context(client)
    )
    assert sorted(r.sample for r in seen) == [0, 1, 2]
    assert result.passed
    assert result.judge is not None
    assert result.judge.agreement == pytest.approx(2 / 3)
    assert result.judge.score_stdev > 0
    assert result.detail == "2/3 runs reached 0.7 (majority needed); agreement 67%"


async def test_one_repair_turn_fixes_bad_json() -> None:
    client, seen = scripted_judge(["not json", verdict(4, 4)])
    result = await judge_case(judge_spec(), CASE, "Paris", context(client))
    assert result.passed
    assert result.judge is not None
    assert result.judge.calls == 2
    repair = seen[1]
    assert [m.role for m in repair.messages] == ["user", "assistant", "user"]
    assert repair.messages[1].content == "not json"
    assert "could not be used" in repair.messages[2].content


async def test_empty_first_reply_still_gets_a_valid_repair_turn() -> None:
    client, seen = scripted_judge(["", verdict(4, 4)])
    await judge_case(judge_spec(), CASE, "Paris", context(client))
    assert seen[1].messages[1].content == "(empty reply)"


async def test_second_bad_reply_becomes_a_failed_score(caplog: pytest.LogCaptureFixture) -> None:
    client, _ = scripted_judge(["nope", "still nope"])
    with caplog.at_level(logging.ERROR):
        result = await judge_case(judge_spec(), CASE, "Paris", context(client))
    assert not result.passed
    assert result.score == 0.0
    assert result.error is not None
    assert "after one repair attempt" in result.error
    assert "case=c1" in caplog.text


class FailingJudge:
    name = "anthropic:judge"

    def __init__(self, error: ProviderError) -> None:
        self.error = error

    async def complete(self, request: LLMRequest) -> Any:
        raise self.error


async def test_non_fatal_provider_error_becomes_a_failed_score() -> None:
    result = await judge_case(
        judge_spec(), CASE, "Paris", context(FailingJudge(ProviderError("HTTP 400: too long")))
    )
    assert not result.passed
    assert result.error == "HTTP 400: too long"


async def test_fatal_provider_error_propagates_with_judge_name() -> None:
    with pytest.raises(ProviderError, match="judge anthropic:judge: HTTP 401") as info:
        await judge_case(
            judge_spec(),
            CASE,
            "Paris",
            context(FailingJudge(ProviderError("HTTP 401: bad key", fatal=True))),
        )
    assert info.value.fatal


async def test_heuristic_fake_judge_produces_valid_verdicts() -> None:
    judge = FakeLLM("judge", heuristic_judge)
    good = await judge_case(judge_spec(), CASE, "Paris", context(judge))
    bad = await judge_case(judge_spec(), CASE, "Lyon, I think", context(judge))
    assert good.score > bad.score
    assert good.judge is not None
    assert good.judge.input_tokens > 0


async def test_apply_scorer_requires_a_judge_for_llm_judge() -> None:
    with pytest.raises(ConfigError, match="judge model is required"):
        await apply_scorer(judge_spec(), CASE, "Paris", None)


def judge_by_sample(replies: dict[int, str], default: str) -> FakeLLM:
    """A fake judge whose reply depends on the repeat index (`sample`)."""
    return FakeLLM("by-sample", lambda request: replies.get(request.sample, default))


async def test_a_failed_repeat_is_counted_and_the_rest_still_decide() -> None:
    judge = judge_by_sample({1: "not json"}, default=verdict(5, 5))
    result = await judge_case(
        judge_spec(repeats=3, aggregate="majority"), CASE, "Paris", context(judge)
    )
    assert result.passed
    assert result.judge is not None
    assert (len(result.judge.runs), result.judge.failed_runs) == (2, 1)
    assert "1 run(s) failed: invalid verdict after one repair attempt" in result.detail


async def test_every_repeat_failing_fails_the_score() -> None:
    judge = judge_by_sample({}, default="not json")
    result = await judge_case(judge_spec(repeats=2), CASE, "Paris", context(judge))
    assert not result.passed
    assert result.judge is None
    assert result.error is not None


async def test_a_score_exactly_at_the_threshold_passes() -> None:
    rubric = Rubric.model_validate(
        {
            "criteria": [
                {"name": "accuracy", "description": "d", "weight": 0.7},
                {"name": "clarity", "description": "d", "weight": 0.3},
            ],
            "pass_threshold": 0.75,
        }
    )
    spec = JudgeSpec.model_validate({"type": "llm_judge", "rubric": rubric.model_dump()})
    client, _ = scripted_judge(
        [verdict(4, 4)]
    )  # 0.7*0.75 + 0.3*0.75 is 0.7499999999999999 in floats
    result = await judge_case(spec, CASE, "Paris", context(client))
    assert result.score == 0.75
    assert result.passed


async def test_unexpected_errors_are_not_hidden() -> None:
    class Buggy:
        name = "fake:buggy"

        async def complete(self, request: LLMRequest) -> Any:
            raise KeyError("a bug")

    with pytest.raises(KeyError):
        await judge_case(judge_spec(), CASE, "Paris", context(Buggy()))
