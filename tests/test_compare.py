from helpers import make_result, make_run

from ljh.compare import compare_runs, render_comparison


def test_classifies_each_case() -> None:
    base = make_run(
        [
            make_result(case_id="kept", passed=True),
            make_result(case_id="broke", passed=True),
            make_result(case_id="fixed", passed=False, score=0.0),
            make_result(case_id="dropped"),
        ],
        run_id="base",
    )
    candidate = make_run(
        [
            make_result(case_id="kept", passed=True),
            make_result(case_id="broke", passed=False, score=0.2),
            make_result(case_id="fixed", passed=True),
            make_result(case_id="new"),
        ],
        run_id="cand",
    )
    comparison = compare_runs(base, candidate)
    statuses = {d.case_id: d.status for d in comparison.diffs}
    assert statuses == {
        "kept": "unchanged",
        "broke": "regression",
        "fixed": "improvement",
        "dropped": "removed",
        "new": "added",
    }
    assert comparison.gate_failed
    broke = comparison.with_status("regression")[0]
    assert (broke.base_score, broke.candidate_score) == (1.0, 0.2)

    text = render_comparison(comparison)
    assert "# Compare `base` → `cand`" in text
    assert "## Regressions (1)" in text
    assert "| fake:a | broke | 1.00 | 0.20 |" in text
    assert "## Improvements (1)" in text
    assert "Unchanged: 1 · Added: 1 · Removed: 1" in text
    assert "| fake:a | 75.0% | 75.0% | +0.0 pts |" in text
    assert "Warning" not in text


def test_no_regressions_and_model_only_in_one_run() -> None:
    base = make_run([make_result(model="fake:a")], suite="one")
    candidate = make_run([make_result(model="fake:a"), make_result(model="fake:b")], suite="two")
    comparison = compare_runs(base, candidate)
    assert not comparison.gate_failed  # an added model is fine
    text = render_comparison(comparison)
    assert "Warning: different suites (one vs two)." in text
    assert "| fake:b | n/a | 100.0% | n/a |" in text
    assert text.count("None.") == 2


def test_a_missing_case_fails_the_gate() -> None:
    base = make_run([make_result(case_id="a"), make_result(case_id="b")])
    candidate = make_run([make_result(case_id="a")])
    comparison = compare_runs(base, candidate)
    assert [d.status for d in comparison.diffs] == ["unchanged", "removed"]
    assert comparison.gate_failed


def test_pass_rate_change_has_the_right_sign() -> None:
    base = make_run([make_result(case_id="a", passed=False), make_result(case_id="b")])
    better = make_run([make_result(case_id="a"), make_result(case_id="b")])
    assert "| fake:a | 50.0% | 100.0% | +50.0 pts |" in render_comparison(
        compare_runs(base, better)
    )
    assert "| fake:a | 100.0% | 50.0% | -50.0 pts |" in render_comparison(
        compare_runs(better, base)
    )
