"""Small builders shared by the tests."""

from __future__ import annotations

from typing import Any

from ljh.results import CaseResult, RunResult
from ljh.suite import Suite


def make_suite(**overrides: Any) -> Suite:
    data: dict[str, Any] = {
        "name": "demo",
        "system_prompt": "Be brief.",
        "default_scorers": [{"type": "contains"}],
        "cases": [
            {"id": "france", "input": "Capital of France?", "reference": "Paris", "tags": ["geo"]},
            {"id": "japan", "input": "Capital of Japan?", "reference": "Tokyo", "tags": ["geo"]},
        ],
    }
    data.update(overrides)
    return Suite.model_validate(data)


def make_result(**overrides: Any) -> CaseResult:
    data: dict[str, Any] = {
        "case_id": "c1",
        "model": "fake:a",
        "tags": ["geo"],
        "input": "question",
        "reference": "answer",
        "output": "answer",
        "passed": True,
        "score": 1.0,
        "scores": [],
        "latency_s": 0.5,
        "input_tokens": 10,
        "output_tokens": 5,
        "cost_usd": 0.001,
        "cached": False,
    }
    data.update(overrides)
    return CaseResult(**data)


def make_run(results: list[CaseResult], **overrides: Any) -> RunResult:
    models = list(dict.fromkeys(r.model for r in results))
    data: dict[str, Any] = {
        "run_id": "20260101T000000Z",
        "suite": "demo",
        "created_at": "2026-01-01T00:00:00+00:00",
        "models": models,
        "judge_model": None,
        "results": results,
    }
    data.update(overrides)
    return RunResult(**data)
