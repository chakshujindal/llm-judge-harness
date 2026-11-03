"""Settings read from environment variables. See .env.example for the full list."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ljh.errors import ConfigError


@dataclass(frozen=True)
class Settings:
    anthropic_model: str = "claude-sonnet-5-5"
    openai_model: str = "gpt-6.1-sol"
    judge_model: str = "anthropic:claude-sonnet-5-5"
    timeout_s: float = 60.0
    max_retries: int = 4
    max_concurrency: int = 8
    requests_per_minute: int = 60
    max_tokens: int = 16000
    cache_path: Path = Path(".ljh_cache.sqlite")
    prices_file: Path | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from `env` (defaults to os.environ); unset keys keep their defaults."""
        env = os.environ if env is None else env
        defaults = cls()
        prices_file = env.get("LJH_PRICES_FILE")
        return cls(
            anthropic_model=env.get("LJH_ANTHROPIC_MODEL", defaults.anthropic_model),
            openai_model=env.get("LJH_OPENAI_MODEL", defaults.openai_model),
            judge_model=env.get("LJH_JUDGE_MODEL", defaults.judge_model),
            timeout_s=_positive_float(env, "LJH_TIMEOUT_S", defaults.timeout_s),
            max_retries=_int_at_least(env, "LJH_MAX_RETRIES", defaults.max_retries, minimum=0),
            max_concurrency=_int_at_least(env, "LJH_MAX_CONCURRENCY", defaults.max_concurrency),
            requests_per_minute=_int_at_least(env, "LJH_RPM", defaults.requests_per_minute),
            max_tokens=_int_at_least(env, "LJH_MAX_TOKENS", defaults.max_tokens),
            cache_path=Path(env.get("LJH_CACHE_PATH", str(defaults.cache_path))),
            prices_file=Path(prices_file) if prices_file else None,
        )


def _int_at_least(env: Mapping[str, str], name: str, default: int, minimum: int = 1) -> int:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _positive_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be > 0, got {value}")
    return value
