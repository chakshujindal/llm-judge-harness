"""Result records written to results.json and read back by `report` and `compare`."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from ljh.errors import HarnessError


class CriterionVerdict(BaseModel):
    """One judge's score for one rubric criterion."""

    name: str
    score: int = Field(ge=1, le=5)
    reasoning: str = Field(max_length=500)


class JudgeRun(BaseModel):
    """One judge call: per-criterion verdicts and their weighted, 0-1 normalised score."""

    score: float
    passed: bool
    criteria: list[CriterionVerdict]


class JudgeDetail(BaseModel):
    judge_model: str
    aggregate: str
    pass_threshold: float
    weights: dict[str, float]
    runs: list[JudgeRun]  # runs that returned a valid verdict
    failed_runs: int = 0  # runs whose call or reply failed
    agreement: float  # share of runs whose pass/fail matches the majority verdict
    score_stdev: float
    calls: int
    cached_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    spent_usd: float | None  # cost of the calls not served from the cache

    @property
    def criterion_means(self) -> dict[str, float]:
        """Mean 1-5 score per criterion across runs."""
        return {
            name: sum(c.score for run in self.runs for c in run.criteria if c.name == name)
            / len(self.runs)
            for name in self.weights
        }


class ScoreResult(BaseModel):
    scorer: str
    passed: bool
    score: float = Field(ge=0, le=1)
    detail: str
    error: str | None = None
    judge: JudgeDetail | None = None


class CaseResult(BaseModel):
    case_id: str
    model: str
    tags: list[str]
    input: str
    reference: str | None
    output: str
    passed: bool
    score: float
    scores: list[ScoreResult]
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    cached: bool
    stop_reason: str | None = None
    error: str | None = None

    @property
    def why(self) -> str:
        """Short reason the case failed (empty if it passed)."""
        if self.error:
            return self.error
        failed = [s for s in self.scores if not s.passed]
        return f"{failed[0].scorer}: {failed[0].detail}" if failed else ""


class RunResult(BaseModel):
    run_id: str
    suite: str
    created_at: str
    models: list[str]
    judge_model: str | None
    results: list[CaseResult]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> RunResult:
        """Load results.json, or <dir>/results.json when given a run directory."""
        file = path / "results.json" if path.is_dir() else path
        try:
            return cls.model_validate_json(file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValidationError) as exc:
            raise HarnessError(f"cannot load run results from {file}: {exc}") from exc
