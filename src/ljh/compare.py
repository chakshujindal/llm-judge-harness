"""Diff two runs case by case, so a CI job can fail on regressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ljh.report.fmt import pct
from ljh.report.markdown import table
from ljh.results import CaseResult, RunResult
from ljh.summary import summarize_models

Status = Literal["regression", "improvement", "unchanged", "added", "removed"]


@dataclass(frozen=True)
class CaseDiff:
    model: str
    case_id: str
    status: Status
    base_score: float | None
    candidate_score: float | None


@dataclass(frozen=True)
class Comparison:
    base: RunResult
    candidate: RunResult
    diffs: list[CaseDiff]

    def with_status(self, status: Status) -> list[CaseDiff]:
        return [d for d in self.diffs if d.status == status]

    @property
    def gate_failed(self) -> bool:
        """CI should fail: a case regressed, or a case disappeared and can no longer be checked."""
        return any(d.status in ("regression", "removed") for d in self.diffs)


def compare_runs(base: RunResult, candidate: RunResult) -> Comparison:
    """Match results on (model, case_id) and classify each pair."""
    before = {(r.model, r.case_id): r for r in base.results}
    after = {(r.model, r.case_id): r for r in candidate.results}
    keys = list(before) + [k for k in after if k not in before]
    diffs = [_diff(key, before.get(key), after.get(key)) for key in keys]
    return Comparison(base=base, candidate=candidate, diffs=diffs)


def _diff(key: tuple[str, str], old: CaseResult | None, new: CaseResult | None) -> CaseDiff:
    model, case_id = key
    if old is None or new is None:
        status: Status = "added" if old is None else "removed"
    elif old.passed and not new.passed:
        status = "regression"
    elif new.passed and not old.passed:
        status = "improvement"
    else:
        status = "unchanged"
    return CaseDiff(
        model=model,
        case_id=case_id,
        status=status,
        base_score=old.score if old else None,
        candidate_score=new.score if new else None,
    )


def render_comparison(comparison: Comparison) -> str:
    base, candidate = comparison.base, comparison.candidate
    lines = [f"# Compare `{base.run_id}` → `{candidate.run_id}`", ""]
    if base.suite != candidate.suite:
        lines += [f"Warning: different suites ({base.suite} vs {candidate.suite}).", ""]
    rates_before = {m.model: m.pass_rate for m in summarize_models(base)}
    rates_after = {m.model: m.pass_rate for m in summarize_models(candidate)}
    models = list(rates_before) + [m for m in rates_after if m not in rates_before]
    lines += table(
        ["Model", "Base pass rate", "Candidate pass rate", "Change"],
        [
            [
                m,
                pct(rates_before.get(m)),
                pct(rates_after.get(m)),
                _delta(m, rates_before, rates_after),
            ]
            for m in models
        ],
    )
    for status, title in (("regression", "Regressions"), ("improvement", "Improvements")):
        found = comparison.with_status(status)
        lines += ["", f"## {title} ({len(found)})", ""]
        if not found:
            lines.append("None.")
            continue
        lines += table(
            ["Model", "Case", "Score before", "Score after"],
            [[d.model, d.case_id, _score(d.base_score), _score(d.candidate_score)] for d in found],
        )
    counts = {s: len(comparison.with_status(s)) for s in ("unchanged", "added", "removed")}
    lines += ["", " · ".join(f"{s.capitalize()}: {n}" for s, n in counts.items()), ""]
    return "\n".join(lines)


def _delta(model: str, before: dict[str, float], after: dict[str, float]) -> str:
    if model not in before or model not in after:
        return "n/a"
    change = (after[model] - before[model]) * 100
    return f"{change:+.1f} pts"


def _score(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"
