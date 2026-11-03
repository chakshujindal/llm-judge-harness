from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def hermetic_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dummy keys so SDK clients can be built, and no real `ant` profile is ever read.

    Tests never reach the network: providers get a mock HTTP transport.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    for name in (
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_PROFILE",
        "ANTHROPIC_CONFIG_DIR",
        "ANTHROPIC_FEDERATION_RULE_ID",
        "ANTHROPIC_BASE_URL",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
