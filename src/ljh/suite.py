"""The suite file format (YAML), validated with Pydantic."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

import jsonschema
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ljh.errors import SuiteError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExactSpec(_Strict):
    """Output must equal `expected` (default: the reference); whitespace runs count as one space."""

    type: Literal["exact"]
    expected: str | None = None
    case_sensitive: bool = False


class ContainsSpec(_Strict):
    """Output must contain the given strings (default: the case reference)."""

    type: Literal["contains"]
    values: list[str] | None = Field(default=None, min_length=1)
    match: Literal["all", "any"] = "all"
    case_sensitive: bool = False


class RegexSpec(_Strict):
    type: Literal["regex"]
    pattern: str
    ignore_case: bool = False

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, pattern: str) -> str:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}") from exc
        return pattern


class JsonSchemaSpec(_Strict):
    """Output must be JSON that validates against `schema` (a JSON Schema document)."""

    type: Literal["json_schema"]
    json_schema: dict[str, Any] = Field(alias="schema")

    @field_validator("json_schema")
    @classmethod
    def _is_valid_schema(cls, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            jsonschema.Draft202012Validator.check_schema(schema)
        except jsonschema.SchemaError as exc:
            raise ValueError(f"invalid JSON Schema: {exc.message}") from exc
        remote = [ref for ref in _refs(schema) if not ref.startswith("#")]
        if remote:
            raise ValueError(f"only local $refs (starting with '#') are supported, got {remote}")
        return schema


def _refs(node: Any) -> list[str]:
    """Every "$ref" value in a JSON Schema document."""
    if isinstance(node, dict):
        own = [node["$ref"]] if isinstance(node.get("$ref"), str) else []
        return own + [ref for value in node.values() for ref in _refs(value)]
    if isinstance(node, list):
        return [ref for item in node for ref in _refs(item)]
    return []


class NumericSpec(_Strict):
    """The first number in the output must be within `tolerance` of `expected` (or reference)."""

    type: Literal["numeric"]
    expected: float | None = None
    tolerance: float = Field(default=0.0, ge=0)


class Criterion(_Strict):
    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    description: str = Field(min_length=1)
    weight: float = Field(default=1.0, gt=0)


class Rubric(_Strict):
    criteria: list[Criterion] = Field(min_length=1)
    pass_threshold: float = Field(default=0.7, ge=0, le=1)

    @field_validator("criteria")
    @classmethod
    def _unique_names(cls, criteria: list[Criterion]) -> list[Criterion]:
        names = [c.name for c in criteria]
        if len(names) != len(set(names)):
            raise ValueError(f"criterion names must be unique, got {names}")
        return criteria


class JudgeSpec(_Strict):
    """Score with an LLM judge against a weighted rubric, optionally repeated N times."""

    type: Literal["llm_judge"]
    rubric: Rubric
    repeats: int = Field(default=1, ge=1, le=10)
    aggregate: Literal["mean", "majority"] = "mean"


ScorerSpec = Annotated[
    ExactSpec | ContainsSpec | RegexSpec | JsonSchemaSpec | NumericSpec | JudgeSpec,
    Field(discriminator="type"),
]


class Case(_Strict):
    model_config = ConfigDict(coerce_numbers_to_str=True)  # `reference: 100` becomes "100"

    id: str = Field(min_length=1)
    input: str = Field(min_length=1)
    reference: str | None = None
    tags: list[str] = Field(default_factory=list)
    scorers: list[ScorerSpec] | None = None  # replaces the suite defaults when set


class Suite(_Strict):
    # Used in the default output folder name, so no slashes or leading dots.
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    description: str = ""
    system_prompt: str | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    default_scorers: list[ScorerSpec] = Field(default_factory=list)
    cases: list[Case] = Field(min_length=1)

    def scorers_for(self, case: Case) -> list[ScorerSpec]:
        return case.scorers if case.scorers is not None else self.default_scorers

    @property
    def uses_judge(self) -> bool:
        return any(
            isinstance(spec, JudgeSpec) for case in self.cases for spec in self.scorers_for(case)
        )

    @model_validator(mode="after")
    def _check_cases(self) -> Suite:
        ids = [case.id for case in self.cases]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate case ids: {duplicates}")
        for case in self.cases:
            scorers = self.scorers_for(case)
            if not scorers:
                raise ValueError(f"case {case.id!r} has no scorers and the suite has no defaults")
            for spec in scorers:
                _check_reference(case, spec)
        return self


def _check_reference(case: Case, spec: ScorerSpec) -> None:
    """Fail at load time, not mid-run, when a scorer needs a reference the case lacks."""
    needs_reference = (isinstance(spec, ExactSpec | NumericSpec) and spec.expected is None) or (
        isinstance(spec, ContainsSpec) and spec.values is None
    )
    if not needs_reference:
        return
    if case.reference is None:
        raise ValueError(f"case {case.id!r}: scorer {spec.type!r} needs a reference answer")
    if isinstance(spec, NumericSpec):
        try:
            float(case.reference)
        except ValueError as exc:
            raise ValueError(
                f"case {case.id!r}: numeric scorer needs a numeric reference, "
                f"got {case.reference!r}"
            ) from exc


def load_suite(path: Path) -> Suite:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise SuiteError(f"cannot read suite {path}: {exc}") from exc
    try:
        return Suite.model_validate(data)
    except ValidationError as exc:
        raise SuiteError(f"invalid suite {path}:\n{exc}") from exc
