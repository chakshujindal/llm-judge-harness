"""End-to-end CLI tests using fake models only."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ljh.cli import app

EXAMPLES = Path(__file__).parent.parent / "examples"
ANSWERS = EXAMPLES / "fake" / "answers.yaml"
runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LJH_CACHE_PATH", str(tmp_path / "cache.sqlite"))
    monkeypatch.chdir(tmp_path)


def run_factual(out: Path, answers: Path = ANSWERS, *extra: str) -> object:
    return runner.invoke(
        app,
        [
            "run",
            str(EXAMPLES / "factual_qa.yaml"),
            "-m", "fake:echo",
            "-m", "fake:strong",
            "-m", "fake:weak",
            "--fake-answers", str(answers),
            "--out", str(out),
            *extra,
        ],
    )  # fmt: skip


def test_run_writes_reports_and_prints_summary(tmp_path: Path) -> None:
    result = run_factual(tmp_path / "run")
    assert result.exit_code == 0, result.output
    assert "| fake:strong | 90.0% | 9/10 |" in result.stdout
    assert "[30/30]" in result.stderr
    data = json.loads((tmp_path / "run" / "results.json").read_text())
    assert len(data["results"]) == 30
    assert (tmp_path / "run" / "report.html").read_text().startswith("<!doctype html>")


def test_run_with_judge_and_default_out_dir(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "-v",
            "run",
            str(EXAMPLES / "json_extraction.yaml"),
            "-m", "fake:strong",
            "--judge", "fake:judge",
            "--fake-answers", str(ANSWERS),
            "--quiet",
            "--no-cache",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "judge: fake:judge" in result.stdout
    assert "[1/5]" not in result.stderr
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert run_dir.name.startswith("json-extraction-")


def test_report_rerenders_from_results(tmp_path: Path) -> None:
    run_factual(tmp_path / "run")
    (tmp_path / "run" / "report.html").unlink()
    result = runner.invoke(app, ["report", str(tmp_path / "run" / "results.json")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "run" / "report.html").exists()
    assert "# factual-qa" in result.stdout


def test_compare_exits_non_zero_on_regressions(tmp_path: Path) -> None:
    run_factual(tmp_path / "base")
    run_factual(tmp_path / "cand", EXAMPLES / "fake" / "answers_v2.yaml")
    diff_file = tmp_path / "diff.md"
    result = runner.invoke(
        app,
        ["compare", str(tmp_path / "base"), str(tmp_path / "cand"), "--markdown", str(diff_file)],
    )
    assert result.exit_code == 1
    assert "| fake:strong | capital-australia | 1.00 | 0.00 |" in result.stdout
    assert "| fake:weak | 40.0% | 50.0% | +10.0 pts |" in result.stdout
    assert "## Improvements (2)" in diff_file.read_text()
    same = runner.invoke(app, ["compare", str(tmp_path / "base"), str(tmp_path / "base")])
    assert same.exit_code == 0


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["-m", "fake:echo", "-m", "FAKE:echo"], "each --model must be a different model"),
        (["-m", "gemini:pro"], "bad model spec"),
        (["-m", "fake:unknown"], "unknown fake model"),
    ],
)
def test_run_errors_exit_2_with_one_line(args: list[str], message: str) -> None:
    result = runner.invoke(app, ["run", str(EXAMPLES / "factual_qa.yaml"), *args])
    assert result.exit_code == 2
    assert f"error: {message}" in result.stderr


def test_missing_judge_key_is_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    result = runner.invoke(
        app,
        [
            "run",
            str(EXAMPLES / "json_extraction.yaml"),
            "-m", "fake:strong",
            "--judge", "openai",
            "--fake-answers", str(ANSWERS),
        ],
    )  # fmt: skip
    assert result.exit_code == 2
    assert "OPENAI_API_KEY" in result.stderr


def test_report_on_missing_file_exits_2(tmp_path: Path) -> None:
    result = runner.invoke(app, ["report", str(tmp_path / "nope.json")])
    assert result.exit_code == 2
    assert "cannot load run results" in result.stderr


def test_unwritable_output_and_cache_paths_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "a-file"
    blocker.write_text("not a folder")
    result = run_factual(blocker / "out")
    assert result.exit_code == 2
    assert result.stderr.strip().splitlines()[-1].startswith("error:")  # after the progress lines

    monkeypatch.setenv("LJH_CACHE_PATH", str(blocker / "cache.sqlite"))
    result = run_factual(tmp_path / "run")
    assert result.exit_code == 2
    assert "cannot open the response cache" in result.stderr
