"""Self-contained HTML report: one sortable table per model with a per-case drill-down."""

from __future__ import annotations

from jinja2 import Environment, PackageLoader

from ljh.report.fmt import money, num, pct, secs
from ljh.results import RunResult
from ljh.summary import results_by_model, summarize_judge, summarize_models, tag_pass_counts


def render_html(run: RunResult) -> str:
    # autoescape: model output is untrusted text and must never become markup.
    env = Environment(loader=PackageLoader("ljh.report", "templates"), autoescape=True)
    env.filters.update(pct=pct, money=money, secs=secs, num=num)
    return env.get_template("report.html.j2").render(
        run=run,
        models=summarize_models(run),
        by_model=results_by_model(run),
        tags=tag_pass_counts(run),
        judge=summarize_judge(run),
    )
