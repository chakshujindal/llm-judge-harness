"""SQLite response cache, so re-running a suite only pays for requests it has not seen."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from ljh.errors import ConfigError
from ljh.llm.base import LLMClient, LLMRequest, LLMResponse

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


def cache_key(model: str, request: LLMRequest) -> str:
    """sha256 over (model, prompt, params): any change to what is sent is a cache miss."""
    payload = json.dumps(
        {"model": model, "request": request.model_dump(mode="json")}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class ResponseCache:
    """Lookups are small local reads, so they run inline on the event loop."""

    def __init__(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(path)
            self._conn.execute(_SCHEMA)
        except (OSError, sqlite3.Error) as exc:
            raise ConfigError(f"cannot open the response cache at {path}: {exc}") from exc

    def get(self, key: str) -> LLMResponse | None:
        row = self._conn.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        return LLMResponse.model_validate_json(row[0]) if row else None

    def put(self, key: str, model: str, response: LLMResponse) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?)",
                (key, model, response.model_dump_json(), datetime.now(UTC).isoformat()),
            )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ResponseCache:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class CachedClient:
    """Serves repeated requests from the cache; only misses reach the wrapped client."""

    def __init__(self, inner: LLMClient, cache: ResponseCache) -> None:
        self.name = inner.name
        self._inner = inner
        self._cache = cache

    async def complete(self, request: LLMRequest) -> LLMResponse:
        key = cache_key(self.name, request)
        hit = self._cache.get(key)
        if hit is not None:
            return hit.model_copy(update={"cached": True})
        response = await self._inner.complete(request)
        self._cache.put(key, self.name, response)
        return response
