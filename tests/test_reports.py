import re
from pathlib import Path

import pytest
from helpers import make_result, make_run

from ljh.errors import HarnessError
from ljh.report import render_html, render_markdown, write_reports
from ljh.report.fmt import clip, money, pct
from ljh.results import CriterionVerdict, JudgeDetail, JudgeRun, RunResult, ScoreResult
from ljh.summary import (
    summarize_judge,
    summarize_models,
    tag_pass_counts,
    total_or_none,
    worst_cases,
)


def judge_detail(
    agreement: float, runs: int = 3, cost: float | None = 0.002, failed_runs: int = 0
) -> JudgeDetail:
    verdicts = [CriterionVerdict(name="accuracy", score=4, reasoning="Mostly <right>.")]
    return JudgeDetail(
        judge_model="anthropic:claude-sonnet-5-5",
        aggregate="majority",
        pass_threshold=0.7,
        weights={"accuracy": 1.0},
        runs=[JudgeRun(score=0.75, passed=True, criteria=verdicts)] * runs,
        failed_runs=failed_runs,
        agreement=agreement,
        score_stdev=0.1,
        calls=runs,
        cached_calls=1,
        input_tokens=300,
        output_tokens=60,
        cost_usd=cost,
        spent_usd=None if cost is None else cost / 2,
    )


def judged(agreement: float, **kwargs: object) -> ScoreResult:
    detail = judge_detail(agreement, **kwargs)  # type: ignore[arg-type]
    return ScoreResult(scorer="llm_judge", passed=True, score=0.75, detail="ok", judge=detail)


def sample_run() -> RunResult:
    """fake:a has 3 cases with skewed latencies; openai:x has no price entry and one error."""
    missing = ScoreResult(scorer="contains", passed=False, score=0.3, detail="missing ['Tokyo']")
    return make_run(
        [
            make_result(
                case_id="france",
                model="fake:a",
                tags=["geo", "easy"],
                scores=[judged(1.0)],
                latency_s=0.5,
            ),
            make_result(
                case_id="japan",
                model="fake:a",
                passed=False,
                score=0.3,
                output="<script>alert(1)</script>",
                scores=[missing],
                latency_s=1.5,
                cached=True,
            ),
            make_result(case_id="peru", model="fake:a", scores=[judged(0.5)], latency_s=10.0),
            make_result(case_id="france", model="openai:x", cost_usd=None, tags=["geo"]),
            make_result(
                case_id="japan",
                model="openai:x",
                passed=False,
                score=0.0,
                error="HTTP 400: too long",
                output="",
                cost_usd=None,
            ),
        ],
        judge_model="anthropic:claude-sonnet-5-5",
    )


def test_model_summaries() -> None:
    a, x = summarize_models(sample_run())
    assert (a.model, a.passed, a.cases) == ("fake:a", 2, 3)
    assert a.pass_rate == pytest.approx(2 / 3)
    assert a.p50_latency_s == 1.5  # the median, not the mean (4.0)
    assert a.cost_usd == pytest.approx(0.003)
    assert a.spent_usd == pytest.approx(0.002)  # the cached case was free
    assert a.cached == 1
    assert (x.errors, x.cost_usd, x.spent_usd) == (1, None, None)  # unknown price: not $0
    empty = summarize_models(make_run([], models=["fake:none"]))[0]
    assert (empty.pass_rate, empty.mean_score, empty.p50_latency_s) == (0.0, 0.0, 0.0)


def test_total_or_none() -> None:
    assert total_or_none([0.5, 0.25]) == 0.75
    assert total_or_none([0.5, None]) is None
    assert total_or_none([]) == 0


def test_tags_worst_cases_and_judge_summary() -> None:
    run = sample_run()
    assert tag_pass_counts(run) == {
        "easy": {"fake:a": (1, 1)},
        "geo": {"fake:a": (2, 3), "openai:x": (1, 2)},
    }
    assert [(r.model, r.case_id, r.score) for r in worst_cases(run)] == [
        ("openai:x", "japan", 0.0),
        ("fake:a", "japan", 0.3),
    ]
    assert len(worst_cases(run, limit=1)) == 1
    judge = summarize_judge(run)
    assert judge is not None
    assert (judge.calls, judge.cached_calls, judge.input_tokens) == (6, 2, 600)
    assert judge.mean_agreement == pytest.approx(0.75)  # mean of 1.0 and 0.5
    assert judge.spent_usd == pytest.approx(0.002)
    assert summarize_judge(make_run([make_result()])) is None


def test_judge_summary_without_repeats_or_prices() -> None:
    single = judged(1.0, runs=1, cost=None, failed_runs=1)
    judge = summarize_judge(make_run([make_result(scores=[single])]))
    assert judge is not None
    assert (judge.mean_agreement, judge.cost_usd, judge.failed_runs) == (None, None, 1)
    text = render_markdown(make_run([make_result(scores=[single])]))
    assert "1 failed run(s)" in text
    assert "mean agreement on repeated cases: n/a" in text
    assert "(cache hits are free): n/a" in text


def test_markdown_summary() -> None:
    text = render_markdown(sample_run())
    assert "# demo" in text
    assert "Run `20260101T000000Z` · 2 model(s)" in text
    assert "| fake:a | 66.7% | 2/3 |" in text
    assert "| geo | 66.7% (2/3) | 50.0% (1/2) |" in text
    assert "| easy | 100.0% (1/1) | - |" in text
    assert "| fake:a | japan | 0.30 | contains: missing ['Tokyo'] |" in text
    assert "HTTP 400: too long" in text
    assert "mean agreement on repeated cases: 75.0%" in text
    assert "(cache hits are free): n/a" in text  # openai:x has no price, so the total is unknown


def test_markdown_spent_total_adds_models_and_judge() -> None:
    run = make_run([make_result(cost_usd=0.01, scores=[judged(1.0)])])
    assert "models + judge (cache hits are free): $0.0110" in render_markdown(run)


def test_markdown_when_everything_passes_and_no_tags() -> None:
    text = render_markdown(make_run([make_result(tags=[])]))
    assert "Every case passed." in text
    assert "Pass rate by tag" not in text
    assert "Judge:" not in text


def test_html_report_is_self_contained_and_escaped() -> None:
    html = render_html(sample_run())
    assert html.startswith("<!doctype html>")
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert not re.search(r"""(src|href)=["']https?://""", html)  # nothing loaded from the network
    assert html.count('<table class="sortable">') == 2
    assert html.count('<tbody class="case"') == 5
    assert 'aria-controls="case-1-2"' in html
    assert "Mostly &lt;right&gt;." in html  # judge reasoning in the drill-down
    assert "agreement 50.0%" in html
    assert "Error: HTTP 400: too long" in html


def test_html_shows_failed_judge_runs() -> None:
    html = render_html(make_run([make_result(scores=[judged(1.0, failed_runs=2)])]))
    assert "2 run(s) failed" in html


def test_write_reports_and_reload(tmp_path: Path) -> None:
    run = sample_run()
    written = write_reports(run, tmp_path / "out")
    assert [p.name for p in written] == ["results.json", "summary.md", "report.html"]
    assert RunResult.load(tmp_path / "out") == run
    assert RunResult.load(tmp_path / "out" / "results.json") == run
    again = write_reports(run, tmp_path / "again", include_results=False)
    assert [p.name for p in again] == ["summary.md", "report.html"]


def test_load_errors(tmp_path: Path) -> None:
    with pytest.raises(HarnessError, match="cannot load run results"):
        RunResult.load(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    with pytest.raises(HarnessError, match="cannot load run results"):
        RunResult.load(bad)
    binary = tmp_path / "binary.json"
    binary.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(HarnessError, match="cannot load run results"):
        RunResult.load(binary)


def test_formatters() -> None:
    assert pct(None) == "n/a"
    assert pct(0.5) == "50.0%"
    assert money(None) == "n/a"
    assert money(0.12345) == "$0.1235"
    assert clip("a\n b", 10) == "a b"
    assert clip("x" * 20, 10) == "x" * 9 + "…"
    assert clip("ok\x1b[31m\x07", 10) == "ok[31m"  # terminal control characters are dropped


def test_why_prefers_error_then_first_failed_score() -> None:
    assert make_result(error="boom").why == "boom"
    assert make_result().why == ""
