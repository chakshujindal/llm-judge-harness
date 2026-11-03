"""Number formatting shared by the Markdown and HTML reports."""

from __future__ import annotations


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def secs(value: float) -> str:
    return f"{value:.2f}s"


def num(value: int) -> str:
    return f"{value:,}"


def clip(text: str, limit: int = 80) -> str:
    """One printable line, at most `limit` characters (safe to print to a terminal)."""
    flat = "".join(ch for ch in " ".join(text.split()) if ch.isprintable())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
