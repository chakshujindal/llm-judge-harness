"""Pure aggregations over a RunResult, shared by the Markdown and HTML reports and `compare`."""

from __future__ import annotations

import statistics
from collections import defaultdict

from pydantic import BaseModel

from ljh.results import CaseResult, JudgeDetail, RunResult


class ModelSummary(BaseModel):
    model: str
    cases: int
    passed: int
    errors: int
    mean_score: float
    p50_latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None when the model has no price entry
    spent_usd: float | None  # cost of calls actually made this run (cache hits are free)
    cached: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.cases if self.cases else 0.0


class JudgeSummary(BaseModel):
    calls: int
    cached_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    spent_usd: float | None
    failed_runs: int
    mean_agreement: float | None  # over cases judged more than once


def total_or_none(values: list[float | None]) -> float | None:
    """Sum of costs, or None when any of them is unknown (no price entry)."""
    known = [v for v in values if v is not None]
    return sum(known) if len(known) == len(values) else None


def results_by_model(run: RunResult) -> dict[str, list[CaseResult]]:
    grouped: dict[str, list[CaseResult]] = {model: [] for model in run.models}
    for result in run.results:
        grouped.setdefault(result.model, []).append(result)
    return grouped


def summarize_models(run: RunResult) -> list[ModelSummary]:
    return [_summarize(model, rows) for model, rows in results_by_model(run).items()]


def _summarize(model: str, rows: list[CaseResult]) -> ModelSummary:
    answered = [r for r in rows if r.error is None]
    return ModelSummary(
        model=model,
        cases=len(rows),
        passed=sum(r.passed for r in rows),
        errors=len(rows) - len(answered),
        mean_score=statistics.fmean(r.score for r in rows) if rows else 0.0,
        p50_latency_s=statistics.median(r.latency_s for r in answered) if answered else 0.0,
        input_tokens=sum(r.input_tokens for r in rows),
        output_tokens=sum(r.output_tokens for r in rows),
        cost_usd=total_or_none([r.cost_usd for r in answered]),
        spent_usd=total_or_none([r.cost_usd for r in answered if not r.cached]),
        cached=sum(r.cached for r in rows),
    )


def tag_pass_counts(run: RunResult) -> dict[str, dict[str, tuple[int, int]]]:
    """{tag: {model: (passed, total)}}, tags sorted alphabetically."""
    counts: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for result in run.results:
        for tag in result.tags:
            cell = counts[tag][result.model]
            cell[0] += result.passed
            cell[1] += 1
    return {
        tag: {model: (cell[0], cell[1]) for model, cell in by_model.items()}
        for tag, by_model in sorted(counts.items())
    }


def worst_cases(run: RunResult, limit: int = 5) -> list[CaseResult]:
    """Failed cases, lowest score first."""
    failed = [r for r in run.results if not r.passed]
    return sorted(failed, key=lambda r: (r.score, r.model, r.case_id))[:limit]


def judge_details(run: RunResult) -> list[JudgeDetail]:
    return [s.judge for r in run.results for s in r.scores if s.judge is not None]


def summarize_judge(run: RunResult) -> JudgeSummary | None:
    details = judge_details(run)
    if not details:
        return None
    repeated = [d.agreement for d in details if len(d.runs) > 1]
    return JudgeSummary(
        calls=sum(d.calls for d in details),
        cached_calls=sum(d.cached_calls for d in details),
        input_tokens=sum(d.input_tokens for d in details),
        output_tokens=sum(d.output_tokens for d in details),
        cost_usd=total_or_none([d.cost_usd for d in details]),
        spent_usd=total_or_none([d.spent_usd for d in details]),
        failed_runs=sum(d.failed_runs for d in details),
        mean_agreement=statistics.fmean(repeated) if repeated else None,
    )
