"""A small, editable price table and the cost estimate built on it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ljh.errors import ConfigError


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens."""

    input: float
    output: float


# List prices (USD per 1M tokens, standard tier, short prompts) checked in November 2026.
# Edit here, or point LJH_PRICES_FILE at a YAML file of the same shape:
#   anthropic:claude-sonnet-5-5: {input: 2.0, output: 10.0}
DEFAULT_PRICES: dict[str, Price] = {
    "anthropic:claude-opus-5-5": Price(4.00, 20.00),
    "anthropic:claude-sonnet-5-5": Price(2.00, 10.00),
    "anthropic:claude-haiku-5-5": Price(0.10, 0.50),
    "openai:gpt-6.1-sol": Price(2.00, 10.00),
    "openai:gpt-6.1-terra": Price(1.00, 5.00),
    "openai:gpt-5.4-mini": Price(0.75, 4.50),
}


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int, prices: dict[str, Price]
) -> float | None:
    """Estimated USD cost of one call, or None when the model has no price entry."""
    if model.startswith("fake:"):
        return 0.0
    price = prices.get(model)
    if price is None:
        return None
    return (input_tokens * price.input + output_tokens * price.output) / 1_000_000


def load_prices(path: Path | None) -> dict[str, Price]:
    """The default table, with entries from `path` (if given) added or overriding."""
    prices = dict(DEFAULT_PRICES)
    if path is None:
        return prices
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        prices.update(
            {str(m): Price(float(p["input"]), float(p["output"])) for m, p in data.items()}
        )
    except (OSError, yaml.YAMLError, AttributeError, KeyError, TypeError, ValueError) as exc:
        # ValueError also covers UnicodeDecodeError and non-numeric prices.
        raise ConfigError(f"invalid prices file {path}: {exc!r}") from exc
    return prices
