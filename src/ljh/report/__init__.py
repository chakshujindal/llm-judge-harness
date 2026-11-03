"""Report writers: results.json, summary.md and report.html for one run."""

from __future__ import annotations

from pathlib import Path

from ljh.report.html import render_html
from ljh.report.markdown import render_markdown
from ljh.results import RunResult

__all__ = ["render_html", "render_markdown", "write_reports"]


def write_reports(run: RunResult, out_dir: Path, *, include_results: bool = True) -> list[Path]:
    """Write the run's files into `out_dir` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if include_results:
        run.save(out_dir / "results.json")
        written.append(out_dir / "results.json")
    (out_dir / "summary.md").write_text(render_markdown(run), encoding="utf-8")
    (out_dir / "report.html").write_text(render_html(run), encoding="utf-8")
    return [*written, out_dir / "summary.md", out_dir / "report.html"]
