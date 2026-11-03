"""Runs every case of a suite against every model, concurrently, and scores the answers."""

from __future__ import annotations

import asyncio
import logging
import statistics
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from ljh.errors import ConfigError, ProviderError, RunAbortedError
from ljh.llm.base import LLMClient, LLMRequest
from ljh.pricing import Price, estimate_cost
from ljh.results import CaseResult, RunResult
from ljh.scoring import JudgeContext, apply_scorer
from ljh.suite import Case, Suite

logger = logging.getLogger(__name__)

OnResult = Callable[[CaseResult, int, int], None]  # (result, done, total)


async def run_suite(
    suite: Suite,
    models: Sequence[LLMClient],
    *,
    judge: JudgeContext | None,
    prices: dict[str, Price],
    max_tokens: int,
    concurrency: int,
    on_result: OnResult | None = None,
    now: datetime | None = None,
) -> RunResult:
    """Evaluate all (model, case) pairs. Results keep suite order, grouped by model."""
    if suite.uses_judge and judge is None:
        raise ConfigError(f"suite {suite.name!r} uses llm_judge, so a judge model is required")
    started = now or datetime.now(UTC)
    semaphore = asyncio.Semaphore(concurrency)
    jobs = [(model, case) for model in models for case in suite.cases]
    done = 0

    async def run_job(model: LLMClient, case: Case) -> CaseResult:
        nonlocal done
        async with semaphore:
            result = await evaluate_case(suite, case, model, judge, prices, max_tokens)
        done += 1
        if on_result:
            on_result(result, done, len(jobs))
        return result

    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(run_job(model, case)) for model, case in jobs]
    except* ProviderError as errors:
        first = errors.exceptions[0]
        raise RunAbortedError(f"run aborted: {first}") from first
    return RunResult(
        run_id=started.strftime("%Y%m%dT%H%M%SZ"),
        suite=suite.name,
        created_at=started.isoformat(timespec="seconds"),
        models=[model.name for model in models],
        judge_model=judge.client.name if judge else None,
        results=[task.result() for task in tasks],
    )


async def evaluate_case(
    suite: Suite,
    case: Case,
    model: LLMClient,
    judge: JudgeContext | None,
    prices: dict[str, Price],
    max_tokens: int,
) -> CaseResult:
    """Ask one model one case and score the answer. Non-fatal call errors become a failed case."""
    request = LLMRequest.single(
        case.input, system=suite.system_prompt, max_tokens=suite.max_tokens or max_tokens
    )
    base = {
        "case_id": case.id,
        "model": model.name,
        "tags": case.tags,
        "input": case.input,
        "reference": case.reference,
    }
    try:
        response = await model.complete(request)
    except ProviderError as exc:
        if exc.fatal:
            raise ProviderError(f"{model.name} on case {case.id!r}: {exc}", fatal=True) from exc
        logger.error("model call failed: model=%s case=%s error=%s", model.name, case.id, exc)
        return CaseResult(
            **base,
            output="",
            passed=False,
            score=0.0,
            scores=[],
            latency_s=0.0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=None,
            cached=False,
            error=str(exc),
        )
    scores = [
        await apply_scorer(spec, case, response.text, judge) for spec in suite.scorers_for(case)
    ]
    return CaseResult(
        **base,
        output=response.text,
        passed=all(s.passed for s in scores),
        score=statistics.fmean(s.score for s in scores),
        scores=scores,
        latency_s=response.latency_s,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=estimate_cost(model.name, response.input_tokens, response.output_tokens, prices),
        cached=response.cached,
        stop_reason=response.stop_reason,
    )
