"""Markdown summary: pass rate per model and per tag, the worst cases, and judge stats."""

from __future__ import annotations

from ljh.report.fmt import clip, money, num, pct, secs
from ljh.results import RunResult
from ljh.summary import (
    summarize_judge,
    summarize_models,
    tag_pass_counts,
    total_or_none,
    worst_cases,
)


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

    return [line(headers), "|" + "---|" * len(headers), *(line(row) for row in rows)]


def render_markdown(run: RunResult, worst: int = 5) -> str:
    models = summarize_models(run)
    cases = len(run.results) // max(len(run.models), 1)
    lines = [
        f"# {run.suite}",
        "",
        f"Run `{run.run_id}` · {len(run.models)} model(s) · {cases} case(s) · "
        f"judge: {run.judge_model or 'none'}",
        "",
        "## Pass rate by model",
        "",
        *table(
            [
                "Model",
                "Pass rate",
                "Passed",
                "Mean score",
                "p50 latency",
                "Tokens in/out",
                "Est. cost",
                "Cache hits",
                "Errors",
            ],
            [
                [
                    m.model,
                    pct(m.pass_rate),
                    f"{m.passed}/{m.cases}",
                    f"{m.mean_score:.2f}",
                    secs(m.p50_latency_s),
                    f"{num(m.input_tokens)}/{num(m.output_tokens)}",
                    money(m.cost_usd),
                    str(m.cached),
                    str(m.errors),
                ]
                for m in models
            ],
        ),
    ]
    tags = tag_pass_counts(run)
    if tags:
        lines += ["", "## Pass rate by tag", ""]
        lines += table(
            ["Tag", *run.models],
            [
                [tag, *(_cell(by_model.get(model)) for model in run.models)]
                for tag, by_model in tags.items()
            ],
        )
    failures = worst_cases(run, worst)
    lines += ["", "## Worst cases", ""]
    if failures:
        lines += table(
            ["Model", "Case", "Case score", "Why", "Output"],
            [
                [r.model, r.case_id, f"{r.score:.2f}", clip(r.why, 90), clip(r.output, 60)]
                for r in failures
            ],
        )
    else:
        lines.append("Every case passed.")
    judge = summarize_judge(run)
    if judge:
        failed = f" · {judge.failed_runs} failed run(s)" if judge.failed_runs else ""
        lines += [
            "",
            f"Judge: {judge.calls} call(s) ({judge.cached_calls} cached){failed} · tokens "
            f"{num(judge.input_tokens)}/{num(judge.output_tokens)} · est. cost "
            f"{money(judge.cost_usd)} · mean agreement on repeated cases: "
            f"{pct(judge.mean_agreement)}",
        ]
    spent = total_or_none([m.spent_usd for m in models] + ([judge.spent_usd] if judge else []))
    lines += ["", f"Spent this run, models + judge (cache hits are free): {money(spent)}", ""]
    return "\n".join(lines)


def _cell(counts: tuple[int, int] | None) -> str:
    if counts is None:
        return "-"
    passed, total = counts
    return f"{pct(passed / total)} ({passed}/{total})"
