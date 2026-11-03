from pathlib import Path
from typing import Any

import pytest
from helpers import make_suite
from pydantic import ValidationError

from ljh.errors import SuiteError
from ljh.suite import ContainsSpec, JsonSchemaSpec, JudgeSpec, NumericSpec, Suite, load_suite

EXAMPLES = Path(__file__).parent.parent / "examples"


def judge_spec(**overrides: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "type": "llm_judge",
        "rubric": {"criteria": [{"name": "accuracy", "description": "Is it right?"}]},
    }
    spec.update(overrides)
    return spec


@pytest.mark.parametrize("name", ["factual_qa.yaml", "json_extraction.yaml"])
def test_example_suites_load(name: str) -> None:
    suite = load_suite(EXAMPLES / name)
    assert suite.cases


def test_case_scorers_replace_defaults() -> None:
    suite = make_suite(
        cases=[
            {"id": "a", "input": "q", "reference": "x"},
            {"id": "b", "input": "q", "reference": "4", "scorers": [{"type": "numeric"}]},
        ]
    )
    assert isinstance(suite.scorers_for(suite.cases[0])[0], ContainsSpec)
    assert isinstance(suite.scorers_for(suite.cases[1])[0], NumericSpec)
    assert not suite.uses_judge


def test_numeric_reference_from_yaml_number_becomes_text(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text(
        "name: n\ncases:\n  - id: a\n    input: q\n    reference: 100\n"
        "    scorers: [{type: numeric}]\n"
    )
    assert load_suite(path).cases[0].reference == "100"


def test_uses_judge_and_schema_alias() -> None:
    suite = make_suite(
        default_scorers=[{"type": "json_schema", "schema": {"type": "object"}}, judge_spec()]
    )
    spec = suite.default_scorers[0]
    assert isinstance(spec, JsonSchemaSpec)
    assert spec.json_schema == {"type": "object"}
    assert isinstance(suite.default_scorers[1], JudgeSpec)
    assert suite.uses_judge


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"cases": []}, "at least 1 item"),
        (
            {"cases": [{"id": "a", "input": "q"}, {"id": "a", "input": "r"}]},
            "duplicate case ids",
        ),
        ({"default_scorers": [], "cases": [{"id": "a", "input": "q"}]}, "has no scorers"),
        ({"cases": [{"id": "a", "input": "q"}]}, "needs a reference answer"),
        (
            {"cases": [{"id": "a", "input": "q", "reference": "ten"}]}
            | {"default_scorers": [{"type": "numeric"}]},
            "needs a numeric reference",
        ),
        ({"default_scorers": [{"type": "regex", "pattern": "("}]}, "invalid regex"),
        (
            {"default_scorers": [{"type": "json_schema", "schema": {"type": "nope"}}]},
            "invalid JSON Schema",
        ),
        ({"default_scorers": [{"type": "bleu"}]}, "does not match any of the expected tags"),
        ({"default_scorers": [{"type": "contains", "values": []}]}, "at least 1 item"),
        (
            {
                "default_scorers": [
                    {"type": "json_schema", "schema": {"$ref": "https://example.com/s.json"}}
                ]
            },
            "only local \\$refs",
        ),
        ({"name": "../escape"}, "String should match pattern"),
        ({"default_scorers": [{"type": "contains", "colour": "red"}]}, "Extra inputs"),
        (
            {
                "default_scorers": [
                    judge_spec(
                        rubric={
                            "criteria": [
                                {"name": "a", "description": "x"},
                                {"name": "a", "description": "y"},
                            ]
                        }
                    )
                ]
            },
            "criterion names must be unique",
        ),
        ({"default_scorers": [judge_spec(repeats=0)]}, "greater than or equal to 1"),
    ],
)
def test_validation_errors(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        make_suite(**overrides)


def test_scorers_with_explicit_expected_need_no_reference() -> None:
    suite = make_suite(
        default_scorers=[
            {"type": "exact", "expected": "x"},
            {"type": "contains", "values": ["x"]},
            {"type": "numeric", "expected": 1},
            {"type": "regex", "pattern": "x"},
        ],
        cases=[{"id": "a", "input": "q"}],
    )
    assert len(suite.default_scorers) == 4


def test_load_suite_wraps_errors(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match="cannot read suite"):
        load_suite(tmp_path / "missing.yaml")
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("name: [unclosed")
    with pytest.raises(SuiteError, match="cannot read suite"):
        load_suite(bad_yaml)
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("name: x\ncases: []\n")
    with pytest.raises(SuiteError, match="invalid suite"):
        load_suite(invalid)


def test_suite_model_is_strict_about_unknown_keys() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        Suite.model_validate({"name": "x", "cases": [{"id": "a", "input": "q"}], "oops": 1})


def test_local_refs_are_allowed() -> None:
    schema = {"$defs": {"n": {"type": "string"}}, "properties": {"name": {"$ref": "#/$defs/n"}}}
    suite = make_suite(default_scorers=[{"type": "json_schema", "schema": schema}])
    assert suite.default_scorers


def test_load_suite_rejects_binary_files(tmp_path: Path) -> None:
    binary = tmp_path / "suite.yaml"
    binary.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(SuiteError, match="cannot read suite"):
        load_suite(binary)
