"""Command line: `ljh run`, `ljh report`, `ljh compare`."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer

from ljh.cache import ResponseCache
from ljh.compare import compare_runs, render_comparison
from ljh.config import Settings
from ljh.errors import ConfigError, HarnessError
from ljh.llm.factory import ClientFactory
from ljh.llm.fake import load_fake_answers
from ljh.pricing import load_prices
from ljh.report import render_markdown, write_reports
from ljh.results import CaseResult, RunResult
from ljh.runner import run_suite
from ljh.scoring import JudgeContext
from ljh.suite import load_suite

app = typer.Typer(
    help="Run a prompt suite across LLMs, score answers with rules and an LLM judge, "
    "and get a pass-rate report.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Log retries and errors.")
    ] = False,
) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def run(
    suite_path: Annotated[Path, typer.Argument(help="Suite YAML file.")],
    models: Annotated[
        list[str],
        typer.Option(
            "--model", "-m", help="provider:model to test; repeat for several (e.g. fake:echo)."
        ),
    ],
    judge: Annotated[
        str | None, typer.Option(help="Judge model for llm_judge scorers [env LJH_JUDGE_MODEL].")
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="Output directory [default: runs/<suite>-<run id>].")
    ] = None,
    concurrency: Annotated[
        int | None, typer.Option(min=1, help="Cases in flight at once [env LJH_MAX_CONCURRENCY].")
    ] = None,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Ignore the response cache.")
    ] = False,
    fake_answers: Annotated[
        Path | None, typer.Option(help="Canned answers YAML for fake:<name> models.")
    ] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="No per-case progress.")] = False,
) -> None:
    """Run every case against every model and write results.json, summary.md and report.html."""
    with _harness_errors():
        settings = Settings.from_env()
        suite = load_suite(suite_path)
        prices = load_prices(settings.prices_file)
        answers = load_fake_answers(fake_answers) if fake_answers else {}
        cache = None if no_cache else ResponseCache(settings.cache_path)
        try:
            factory = ClientFactory(settings, cache=cache, fake_answers=answers)
            clients = [factory.build(spec) for spec in models]
            names = [client.name for client in clients]  # "anthropic" == "anthropic:<default>"
            if len(set(names)) != len(names):
                raise ConfigError(f"each --model must be a different model, got {names}")
            judge_ctx = None
            if suite.uses_judge:
                judge_client = factory.build(judge or settings.judge_model)
                judge_ctx = JudgeContext(judge_client, prices, settings.max_tokens)
            result = asyncio.run(
                run_suite(
                    suite,
                    clients,
                    judge=judge_ctx,
                    prices=prices,
                    max_tokens=settings.max_tokens,
                    concurrency=concurrency or settings.max_concurrency,
                    on_result=None if quiet else _print_progress,
                )
            )
        finally:
            if cache:
                cache.close()
        out_dir = out or Path("runs") / f"{suite.name}-{result.run_id}"
        written = write_reports(result, out_dir)
        typer.echo(render_markdown(result))
        typer.echo(f"Wrote {', '.join(str(p) for p in written)}", err=True)


@app.command()
def report(
    results: Annotated[Path, typer.Argument(help="results.json, or a run directory.")],
    out: Annotated[
        Path | None, typer.Option(help="Output directory [default: next to it].")
    ] = None,
) -> None:
    """Re-render summary.md and report.html from a saved results.json."""
    with _harness_errors():
        run_result = RunResult.load(results)
        out_dir = out or (results if results.is_dir() else results.parent)
        written = write_reports(run_result, out_dir, include_results=False)
        typer.echo(render_markdown(run_result))
        typer.echo(f"Wrote {', '.join(str(p) for p in written)}", err=True)


@app.command()
def compare(
    base: Annotated[Path, typer.Argument(help="Baseline results.json or run directory.")],
    candidate: Annotated[Path, typer.Argument(help="Candidate results.json or run directory.")],
    markdown: Annotated[Path | None, typer.Option(help="Also write the diff to this file.")] = None,
) -> None:
    """Diff two runs per case. Exits 1 when a case regressed (passed, now fails) or went missing."""
    with _harness_errors():
        comparison = compare_runs(RunResult.load(base), RunResult.load(candidate))
        text = render_comparison(comparison)
        typer.echo(text)
        if markdown:
            markdown.write_text(text, encoding="utf-8")
    if comparison.gate_failed:
        raise typer.Exit(1)


def _print_progress(result: CaseResult, done: int, total: int) -> None:
    status = "PASS" if result.passed else ("ERR " if result.error else "FAIL")
    width = len(str(total))
    typer.echo(
        f"[{done:>{width}}/{total}] {status} {result.score:.2f}  {result.model}  {result.case_id}",
        err=True,
    )


@contextmanager
def _harness_errors() -> Iterator[None]:
    """The CLI boundary: expected errors become one line on stderr and exit code 2."""
    try:
        yield
    except (HarnessError, OSError) as exc:  # OSError: e.g. an output folder we cannot write
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
