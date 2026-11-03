"""LLM-as-judge scorer: a weighted rubric, validated JSON verdicts and optional repeats."""

from __future__ import annotations

import asyncio
import logging
import statistics
from dataclasses import dataclass

from pydantic import BaseModel

from ljh.errors import JudgeOutputError, ProviderError
from ljh.llm.base import LLMClient, LLMRequest, LLMResponse, Message
from ljh.pricing import Price, estimate_cost
from ljh.results import CriterionVerdict, JudgeDetail, JudgeRun, ScoreResult
from ljh.suite import Case, JudgeSpec, Rubric

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = """\
You are a strict, impartial evaluator of AI assistant responses.
Score the response on each rubric criterion with an integer from 1 (fails it) to 5 (fully meets it).
When a reference answer is given, treat it as ground truth.
Reply with only a JSON object, with no prose and no code fences, in exactly this shape:
{"criteria": [{"name": "<criterion name>", "score": <1-5>, "reasoning": "<one short sentence>"}]}
Include every criterion exactly once, using the names given."""

REPAIR_PROMPT = """\
Your reply could not be used: {error}
Reply again with only the corrected JSON object."""


class _Verdict(BaseModel):
    criteria: list[CriterionVerdict]


@dataclass(frozen=True)
class JudgeContext:
    client: LLMClient
    prices: dict[str, Price]
    max_tokens: int


def build_judge_prompt(rubric: Rubric, case_input: str, reference: str | None, output: str) -> str:
    criteria = "\n".join(
        f'<criterion name="{c.name}" weight="{c.weight:g}">{c.description}</criterion>'
        for c in rubric.criteria
    )
    reference_block = (
        f"<reference>\n{reference}\n</reference>\n\n"
        if reference is not None
        else "There is no reference answer; judge against the rubric alone.\n\n"
    )
    return (
        "Evaluate the response to the input using this rubric.\n\n"
        f"<rubric>\n{criteria}\n</rubric>\n\n"
        f"<input>\n{case_input}\n</input>\n\n"
        f"{reference_block}"
        f"<response>\n{output}\n</response>"
    )


def parse_verdict(text: str, rubric: Rubric) -> list[CriterionVerdict]:
    """Validate the judge's JSON; raises ValueError (incl. pydantic's) when it does not fit."""
    start, end = text.find("{"), text.rfind("}")
    verdict = _Verdict.model_validate_json(text[start : end + 1] if start != -1 else text)
    got = sorted(c.name for c in verdict.criteria)
    expected = sorted(c.name for c in rubric.criteria)
    if got != expected:
        raise ValueError(f"criteria must be exactly {expected}, got {got}")
    return verdict.criteria


def weighted_score(criteria: list[CriterionVerdict], rubric: Rubric) -> float:
    """Weighted mean of the 1-5 scores, normalised to 0-1."""
    weights = {c.name: c.weight for c in rubric.criteria}
    total = sum(weights[c.name] * (c.score - 1) / 4 for c in criteria)
    # Rounding keeps an exact 0.75 from becoming 0.7499999999999999 and missing a 0.75 threshold.
    return round(total / sum(weights.values()), 9)


def aggregate(runs: list[JudgeRun], method: str, threshold: float) -> tuple[bool, float, float]:
    """Combine repeated runs into (passed, mean score, agreement).

    mean:     pass when the mean score reaches the threshold.
    majority: pass when more than half of the runs passed on their own.
    Agreement is the share of runs whose verdict matches the majority verdict (1.0 = unanimous).
    """
    mean = statistics.fmean(r.score for r in runs)
    votes = sum(r.passed for r in runs)
    passed = mean >= threshold if method == "mean" else votes * 2 > len(runs)
    agreement = max(votes, len(runs) - votes) / len(runs)
    return passed, mean, agreement


async def judge_case(spec: JudgeSpec, case: Case, output: str, ctx: JudgeContext) -> ScoreResult:
    """Run the judge `spec.repeats` times and aggregate the runs that returned a valid verdict.

    A run fails when its call fails or its reply is still invalid after one repair turn.
    The score fails outright only when every run failed; partial failures are counted.
    """
    prompt = build_judge_prompt(spec.rubric, case.input, case.reference, output)
    outcomes = await asyncio.gather(
        *(_judge_once(prompt, spec.rubric, sample, ctx) for sample in range(spec.repeats)),
        return_exceptions=True,
    )
    failures = [o for o in outcomes if isinstance(o, BaseException)]
    successes = [o for o in outcomes if not isinstance(o, BaseException)]
    for failure in failures:
        if isinstance(failure, ProviderError) and failure.fatal:
            raise ProviderError(f"judge {ctx.client.name}: {failure}", fatal=True) from failure
        if not isinstance(failure, JudgeOutputError | ProviderError):
            raise failure  # a bug, not a judge problem: let it surface
    if failures:
        logger.error(
            "judge failed: judge=%s case=%s failed_runs=%d/%d error=%s",
            ctx.client.name,
            case.id,
            len(failures),
            spec.repeats,
            failures[0],
        )
    if not successes:
        return ScoreResult(
            scorer=spec.type, passed=False, score=0.0, detail="judge failed", error=str(failures[0])
        )
    runs = [run for run, _ in successes]
    responses = [response for _, calls in successes for response in calls]
    passed, mean, agreement = aggregate(runs, spec.aggregate, spec.rubric.pass_threshold)
    detail = JudgeDetail(
        judge_model=ctx.client.name,
        aggregate=spec.aggregate,
        pass_threshold=spec.rubric.pass_threshold,
        weights={c.name: c.weight for c in spec.rubric.criteria},
        runs=runs,
        failed_runs=len(failures),
        agreement=agreement,
        score_stdev=statistics.pstdev(r.score for r in runs),
        calls=len(responses),
        cached_calls=sum(r.cached for r in responses),
        input_tokens=sum(r.input_tokens for r in responses),
        output_tokens=sum(r.output_tokens for r in responses),
        cost_usd=_cost(responses, ctx),
        spent_usd=_cost([r for r in responses if not r.cached], ctx),
    )
    return ScoreResult(
        scorer=spec.type,
        passed=passed,
        score=mean,
        detail=_describe(spec, runs, mean, agreement, failures),
        judge=detail,
    )


def _describe(
    spec: JudgeSpec,
    runs: list[JudgeRun],
    mean: float,
    agreement: float,
    failures: list[BaseException],
) -> str:
    threshold = spec.rubric.pass_threshold
    if spec.aggregate == "majority":
        votes = sum(r.passed for r in runs)
        text = f"{votes}/{len(runs)} runs reached {threshold:g} (majority needed)"
    else:
        text = f"mean {mean:.2f} vs threshold {threshold:g} over {len(runs)} run(s)"
    text += f"; agreement {agreement:.0%}"
    if failures:
        text += f"; {len(failures)} run(s) failed: {failures[0]}"
    return text


def _cost(responses: list[LLMResponse], ctx: JudgeContext) -> float | None:
    input_tokens = sum(r.input_tokens for r in responses)
    output_tokens = sum(r.output_tokens for r in responses)
    return estimate_cost(ctx.client.name, input_tokens, output_tokens, ctx.prices)


async def _judge_once(
    prompt: str, rubric: Rubric, sample: int, ctx: JudgeContext
) -> tuple[JudgeRun, list[LLMResponse]]:
    """One judge verdict, with a single repair turn if the first reply is not valid."""
    request = LLMRequest.single(
        prompt, system=JUDGE_SYSTEM, max_tokens=ctx.max_tokens, sample=sample
    )
    first = await ctx.client.complete(request)
    try:
        return _run(parse_verdict(first.text, rubric), rubric), [first]
    except ValueError as exc:
        problem = _short_error(exc)
    repair = request.model_copy(
        update={
            "messages": (
                *request.messages,
                Message(role="assistant", content=first.text or "(empty reply)"),
                Message(role="user", content=REPAIR_PROMPT.format(error=problem)),
            )
        }
    )
    second = await ctx.client.complete(repair)
    try:
        return _run(parse_verdict(second.text, rubric), rubric), [first, second]
    except ValueError as exc:
        raise JudgeOutputError(
            f"invalid verdict after one repair attempt: {_short_error(exc)}"
        ) from exc


def _run(criteria: list[CriterionVerdict], rubric: Rubric) -> JudgeRun:
    score = weighted_score(criteria, rubric)
    return JudgeRun(score=score, passed=score >= rubric.pass_threshold, criteria=criteria)


def _short_error(exc: ValueError) -> str:
    return " ".join(str(exc).split())[:300]
