"""Rule scorers: pure functions of (spec, output, reference) -> ScoreResult."""

from __future__ import annotations

import json
import re

import jsonschema

from ljh.results import ScoreResult
from ljh.suite import ContainsSpec, ExactSpec, JsonSchemaSpec, NumericSpec, RegexSpec

RuleSpec = ExactSpec | ContainsSpec | RegexSpec | JsonSchemaSpec | NumericSpec

# A number not glued to a word ("GPT-4" gives 4, not -4); commas only as thousands separators.
_NUMBER = re.compile(r"(?<![\w.])-?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)")
_FENCE = re.compile(r"^```[a-zA-Z]*\n(.*?)\n?```$", flags=re.DOTALL)


def score_rule(spec: RuleSpec, output: str, reference: str | None) -> ScoreResult:
    if isinstance(spec, ExactSpec):
        return _exact(spec, output, reference)
    if isinstance(spec, ContainsSpec):
        return _contains(spec, output, reference)
    if isinstance(spec, RegexSpec):
        return _regex(spec, output)
    if isinstance(spec, JsonSchemaSpec):
        return _json_schema(spec, output)
    return _numeric(spec, output, reference)


def strip_code_fence(text: str) -> str:
    """Remove a surrounding ```lang ... ``` block, which models often add around JSON."""
    text = text.strip()
    match = _FENCE.match(text)
    return match.group(1).strip() if match else text


def _result(spec: RuleSpec, passed: bool, detail: str, score: float | None = None) -> ScoreResult:
    if score is None:
        score = 1.0 if passed else 0.0
    return ScoreResult(scorer=spec.type, passed=passed, score=score, detail=detail)


def _normalize(text: str, case_sensitive: bool) -> str:
    text = " ".join(text.split())
    return text if case_sensitive else text.casefold()


def _exact(spec: ExactSpec, output: str, reference: str | None) -> ScoreResult:
    expected = spec.expected if spec.expected is not None else reference or ""
    passed = _normalize(output, spec.case_sensitive) == _normalize(expected, spec.case_sensitive)
    return _result(spec, passed, f"expected exactly {expected!r}")


def _contains(spec: ContainsSpec, output: str, reference: str | None) -> ScoreResult:
    values = spec.values if spec.values is not None else [reference or ""]
    haystack = _normalize(output, spec.case_sensitive)
    found = [v for v in values if _normalize(v, spec.case_sensitive) in haystack]
    missing = [v for v in values if v not in found]
    if spec.match == "any":
        passed = bool(found)
        return _result(spec, passed, f"found {found}" if passed else f"none of {values} found")
    detail = f"all {len(values)} found" if not missing else f"missing {missing}"
    return _result(spec, not missing, detail, score=len(found) / len(values))


def _regex(spec: RegexSpec, output: str) -> ScoreResult:
    flags = re.IGNORECASE if spec.ignore_case else 0
    match = re.search(spec.pattern, output, flags)
    if match:
        return _result(spec, True, f"matched {match.group(0)!r}")
    return _result(spec, False, f"no match for /{spec.pattern}/")


def _json_schema(spec: JsonSchemaSpec, output: str) -> ScoreResult:
    try:
        data = json.loads(strip_code_fence(output))
    except json.JSONDecodeError as exc:
        return _result(spec, False, f"output is not valid JSON ({exc.msg})")
    except ValueError as exc:  # e.g. an integer too long to convert
        return _result(spec, False, f"output is not valid JSON ({exc})")
    validator = jsonschema.Draft202012Validator(spec.json_schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    if not errors:
        return _result(spec, True, "valid against schema")
    messages = [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors[:3]]
    return _result(spec, False, "; ".join(messages))


def _numeric(spec: NumericSpec, output: str, reference: str | None) -> ScoreResult:
    expected = spec.expected if spec.expected is not None else float(reference or "nan")
    found = _first_number(output)
    if found is None:
        return _result(spec, False, f"no number in output (expected {expected:g})")
    passed = abs(found - expected) <= spec.tolerance
    return _result(spec, passed, f"found {found:g}, expected {expected:g} ± {spec.tolerance:g}")


def _first_number(text: str) -> float | None:
    match = _NUMBER.search(text)
    return float(match.group(0).replace(",", "")) if match else None
