"""Scorer dispatch: rule scorers are pure functions; the judge needs an LLM client."""

from __future__ import annotations

from ljh.errors import ConfigError
from ljh.results import ScoreResult
from ljh.scoring.judge import JudgeContext, judge_case
from ljh.scoring.rules import score_rule
from ljh.suite import Case, JudgeSpec, ScorerSpec

__all__ = ["JudgeContext", "apply_scorer"]


async def apply_scorer(
    spec: ScorerSpec, case: Case, output: str, judge: JudgeContext | None
) -> ScoreResult:
    if isinstance(spec, JudgeSpec):
        if judge is None:
            raise ConfigError("this suite uses llm_judge, so a judge model is required")
        return await judge_case(spec, case, output, judge)
    return score_rule(spec, output, case.reference)
