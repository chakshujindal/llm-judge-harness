"""Deterministic fake models for tests and the no-API-key demo mode.

Built-in fakes:
  fake:echo   repeats the prompt back.
  fake:judge  a stand-in judge that scores by word overlap with the reference answer.
Any other fake:<name> answers from a canned-answers YAML file (see examples/fake/).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import yaml

from ljh.errors import ConfigError
from ljh.llm.base import LLMRequest, LLMResponse

Responder = Callable[[LLMRequest], str]

FALLBACK_ANSWER = "I don't know."
_WORD = re.compile(r"\w+")


class FakeLLM:
    def __init__(self, model: str, responder: Responder) -> None:
        self.name = f"fake:{model}"
        self._responder = responder

    async def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        text = self._responder(request)
        prompt = " ".join([request.system or "", *(m.content for m in request.messages)])
        return LLMResponse(
            text=text,
            input_tokens=_rough_token_count(prompt),
            output_tokens=_rough_token_count(text),
            latency_s=time.perf_counter() - started,
            stop_reason="end_turn",
        )


def echo(request: LLMRequest) -> str:
    return request.messages[-1].content


def scripted(answers: Mapping[str, str]) -> Responder:
    """Reply with the first canned answer whose key appears in the last user message."""

    def respond(request: LLMRequest) -> str:
        prompt = request.messages[-1].content
        for key, answer in answers.items():
            if key in prompt:
                return answer
        return FALLBACK_ANSWER

    return respond


def heuristic_judge(request: LLMRequest) -> str:
    """Score each rubric criterion 1-5 from word overlap between response and reference.

    The `sample` index adds a deterministic +/-1 wobble so repeated judging shows
    realistic (imperfect) agreement.
    """
    prompt = request.messages[0].content
    names = re.findall(r'<criterion name="([^"]+)"', prompt)
    reference = _section(prompt, "reference")
    response = _section(prompt, "response")
    ref_words, resp_words = _words(reference), _words(response)
    shared = len(ref_words & resp_words)
    if ref_words and resp_words:
        # F1 of word overlap: missing reference words and invented extra words both cost.
        precision, recall = shared / len(resp_words), shared / len(ref_words)
        overlap = 2 * precision * recall / (precision + recall) if shared else 0.0
        extra = len(resp_words) - shared
        why = f"{shared} of {len(ref_words)} reference terms present, {extra} extra."
    else:
        overlap = 0.6 if resp_words else 0.0
        why = "No reference to compare with; judged on giving an answer at all."
    base = 1 + round(4 * overlap)
    criteria = [
        {
            "name": name,
            "score": _wobble(base, f"{response}|{name}|{request.sample}"),
            "reasoning": why,
        }
        for name in names
    ]
    return json.dumps({"criteria": criteria})


def load_fake_answers(path: Path) -> dict[str, dict[str, str]]:
    """Read {model_name: {prompt_substring: answer}} from a YAML file."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read fake answers file {path}: {exc}") from exc
    if not isinstance(data, dict) or not all(isinstance(v, dict) for v in data.values()):
        raise ConfigError(f"{path}: expected a mapping of model name -> {{prompt text: answer}}")
    return {
        str(model): {str(k): str(v) for k, v in answers.items()} for model, answers in data.items()
    }


def build_fake(model: str, answers: Mapping[str, Mapping[str, str]]) -> FakeLLM:
    if model == "echo":
        return FakeLLM(model, echo)
    if model == "judge":
        return FakeLLM(model, heuristic_judge)
    if model in answers:
        return FakeLLM(model, scripted(answers[model]))
    known = ", ".join(["echo", "judge", *answers])
    raise ConfigError(f"unknown fake model 'fake:{model}' (known: {known}; see --fake-answers)")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _section(prompt: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", prompt, flags=re.DOTALL)
    return match.group(1) if match else ""


def _wobble(score: int, seed: str) -> int:
    shift = hashlib.sha256(seed.encode()).digest()[0] % 3 - 1  # -1, 0 or +1
    return min(5, max(1, score + shift))


def _rough_token_count(text: str) -> int:
    return max(1, len(text) // 4) if text else 0
