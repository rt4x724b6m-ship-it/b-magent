from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from b_magent.web.models import WebSearchResult


class ServerWebSearchStore:
    """TTL-aware persistent cache for server-side web evidence."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()

    def lookup(self, queries: list[str]) -> list[WebSearchResult]:
        query_key = _query_key(queries)
        now = datetime.now(timezone.utc)
        for payload in reversed(self._read_payloads()):
            if payload.get("query_key") != query_key:
                continue
            expires_at = _parse_time(str(payload.get("expires_at", "")))
            if expires_at is None or expires_at <= now:
                return []
            results = payload.get("results", [])
            if not isinstance(results, list):
                return []
            return [WebSearchResult.from_dict(item) for item in results if isinstance(item, dict)]
        return []

    def store(
        self,
        question: str,
        queries: list[str],
        results: list[WebSearchResult],
        ttl_seconds: int,
    ) -> None:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        payload = {
            "question": question,
            "queries": queries,
            "query_key": _query_key(queries),
            "created_at": now.isoformat().replace("+00:00", "Z"),
            "expires_at": (now + timedelta(seconds=max(ttl_seconds, 1)))
            .isoformat()
            .replace("+00:00", "Z"),
            "results": [result.to_dict() for result in results],
        }
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _read_payloads(self) -> list[dict[str, object]]:
        payloads: list[dict[str, object]] = []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
        return payloads


def _query_key(queries: list[str]) -> str:
    return " | ".join(sorted({" ".join(query.lower().split()) for query in queries if query.strip()}))


def _parse_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
