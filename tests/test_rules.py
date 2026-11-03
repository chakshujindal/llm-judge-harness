from typing import Any

import pytest
from pydantic import TypeAdapter

from ljh.scoring.rules import RuleSpec, score_rule, strip_code_fence

ADAPTER: TypeAdapter[RuleSpec] = TypeAdapter(RuleSpec)


def spec(**data: Any) -> RuleSpec:
    return ADAPTER.validate_python(data)


@pytest.mark.parametrize(
    ("data", "output", "reference", "passed"),
    [
        ({"type": "exact"}, "  paris ", "Paris", True),
        ({"type": "exact", "case_sensitive": True}, "paris", "Paris", False),
        ({"type": "exact", "expected": "Au"}, "Au", None, True),
        ({"type": "exact"}, "Paris.", "Paris", False),
        ({"type": "regex", "pattern": r"\b1969\b"}, "In 1969.", None, True),
        ({"type": "regex", "pattern": "^yes", "ignore_case": True}, "YES sir", None, True),
        ({"type": "regex", "pattern": "^yes"}, "no", None, False),
    ],
)
def test_exact_and_regex(
    data: dict[str, Any], output: str, reference: str | None, passed: bool
) -> None:
    result = score_rule(spec(**data), output, reference)
    assert result.passed is passed
    assert result.score == (1.0 if passed else 0.0)
    assert result.scorer == data["type"]


def test_contains_reference_and_partial_credit() -> None:
    assert score_rule(spec(type="contains"), "It is PARIS.", "Paris").passed
    partial = score_rule(spec(type="contains", values=["red", "green", "blue"]), "red, blue", None)
    assert not partial.passed
    assert partial.score == pytest.approx(2 / 3)
    assert "missing ['green']" in partial.detail
    full = score_rule(spec(type="contains", values=["a", "b"]), "a b", None)
    assert full.detail == "all 2 found"


def test_contains_any_and_case_sensitive() -> None:
    any_spec = spec(type="contains", values=["x", "y"], match="any")
    assert score_rule(any_spec, "only y", None).passed
    missed = score_rule(any_spec, "none", None)
    assert not missed.passed
    assert "none of" in missed.detail
    assert not score_rule(spec(type="contains", case_sensitive=True), "paris", "Paris").passed


def test_json_schema_scorer() -> None:
    schema_spec = spec(
        type="json_schema",
        schema={
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        },
    )
    assert score_rule(schema_spec, '{"name": "Ana"}', None).passed
    assert score_rule(schema_spec, '```json\n{"name": "Ana"}\n```', None).passed
    not_json = score_rule(schema_spec, "Sure! {name: Ana}", None)
    assert not not_json.passed
    assert "not valid JSON" in not_json.detail
    invalid = score_rule(schema_spec, '{"age": "old"}', None)
    assert not invalid.passed
    assert "age: 'old' is not of type 'integer'" in invalid.detail
    assert "<root>: 'name' is a required property" in invalid.detail


@pytest.mark.parametrize(
    ("data", "output", "reference", "passed"),
    [
        ({"type": "numeric"}, "Water boils at 100 °C.", "100", True),
        ({"type": "numeric", "tolerance": 1000}, "About 299,792 km/s", "300000", True),
        ({"type": "numeric"}, "It is 212 degrees.", "100", False),
        ({"type": "numeric", "expected": -0.5}, "Roughly -.5 units", None, True),
        ({"type": "numeric", "expected": 3.14, "tolerance": 0.01}, "pi is 3.1416", None, True),
    ],
)
def test_numeric(data: dict[str, Any], output: str, reference: str | None, passed: bool) -> None:
    assert score_rule(spec(**data), output, reference).passed is passed


def test_numeric_without_a_number() -> None:
    result = score_rule(spec(type="numeric"), "about a hundred", "100")
    assert not result.passed
    assert result.detail == "no number in output (expected 100)"


def test_strip_code_fence() -> None:
    assert strip_code_fence("```\n[1]\n```") == "[1]"
    assert strip_code_fence('  {"a": 1}  ') == '{"a": 1}'


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("GPT-4 says the answer is 4", 4.0),
        ("COVID-19 began in 2019", 19.0),
        ("1,2,3", 1.0),
        ("It costs 1,250.50 dollars", 1250.5),
        ("abc123 then 7", 7.0),
    ],
)
def test_first_number_ignores_hyphens_and_odd_commas(output: str, expected: float) -> None:
    result = score_rule(spec(type="numeric", expected=expected), output, None)
    assert result.passed, result.detail


def test_json_with_a_huge_integer_is_a_failed_score_not_a_crash() -> None:
    schema_spec = spec(type="json_schema", schema={"type": "object"})
    result = score_rule(schema_spec, '{"n": ' + "9" * 5000 + "}", None)
    assert not result.passed
    assert "not valid JSON" in result.detail
