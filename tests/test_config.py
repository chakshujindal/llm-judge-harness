from pathlib import Path

import pytest

from ljh.config import Settings
from ljh.errors import ConfigError


def test_defaults_when_env_is_empty() -> None:
    settings = Settings.from_env({})
    assert settings == Settings()
    assert settings.anthropic_model == "claude-sonnet-5-5"
    assert settings.prices_file is None


def test_reads_every_variable() -> None:
    settings = Settings.from_env(
        {
            "LJH_ANTHROPIC_MODEL": "claude-opus-5-5",
            "LJH_OPENAI_MODEL": "gpt-6.1-terra",
            "LJH_JUDGE_MODEL": "openai",
            "LJH_TIMEOUT_S": "12.5",
            "LJH_MAX_RETRIES": "0",
            "LJH_MAX_CONCURRENCY": "3",
            "LJH_RPM": "30",
            "LJH_MAX_TOKENS": "2048",
            "LJH_CACHE_PATH": "/tmp/x.sqlite",
            "LJH_PRICES_FILE": "prices.yaml",
        }
    )
    assert settings.anthropic_model == "claude-opus-5-5"
    assert settings.openai_model == "gpt-6.1-terra"
    assert settings.judge_model == "openai"
    assert settings.timeout_s == 12.5
    assert settings.max_retries == 0
    assert settings.max_concurrency == 3
    assert settings.requests_per_minute == 30
    assert settings.max_tokens == 2048
    assert settings.cache_path == Path("/tmp/x.sqlite")
    assert settings.prices_file == Path("prices.yaml")


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"LJH_RPM": "fast"}, "LJH_RPM must be an integer"),
        ({"LJH_MAX_CONCURRENCY": "0"}, "LJH_MAX_CONCURRENCY must be >= 1"),
        ({"LJH_MAX_RETRIES": "-1"}, "LJH_MAX_RETRIES must be >= 0"),
        ({"LJH_TIMEOUT_S": "soon"}, "LJH_TIMEOUT_S must be a number"),
        ({"LJH_TIMEOUT_S": "0"}, "LJH_TIMEOUT_S must be > 0"),
    ],
)
def test_rejects_bad_values(env: dict[str, str], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        Settings.from_env(env)


def test_uses_os_environ_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LJH_MAX_TOKENS", "77")
    assert Settings.from_env().max_tokens == 77
