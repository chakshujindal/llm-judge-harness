"""Exception types. Every error the harness raises on purpose derives from HarnessError."""

from __future__ import annotations


class HarnessError(Exception):
    """Base class: the CLI turns these into a one-line message and a non-zero exit."""


class ConfigError(HarnessError):
    """Bad environment variable, CLI option or model spec."""


class SuiteError(HarnessError):
    """The suite YAML file is missing, malformed or fails validation."""


class ProviderError(HarnessError):
    """A model call failed and retrying will not help.

    `fatal` marks errors that would hit every case the same way (bad API key,
    unknown model), so the runner aborts instead of recording one failure per case.
    """

    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


class RetryableError(ProviderError):
    """A transient failure (429, 5xx, network, timeout) that is safe to retry."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class JudgeOutputError(HarnessError):
    """The judge reply was not valid JSON for the rubric, even after one repair attempt."""


class RunAbortedError(HarnessError):
    """A fatal provider error stopped the run."""
