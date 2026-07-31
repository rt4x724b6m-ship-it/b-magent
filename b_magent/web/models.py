from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str = ""
    content: str = ""
    source: str = ""
    retrieved_at: str = ""
    image_urls: list[str] = field(default_factory=list)
    local_image_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["retrieved_at"] = self.retrieved_at or utc_now()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WebSearchResult":
        return cls(
            title=str(payload.get("title", "")),
            url=str(payload.get("url", "")),
            snippet=str(payload.get("snippet", "")),
            content=str(payload.get("content", "")),
            source=str(payload.get("source", "")),
            retrieved_at=str(payload.get("retrieved_at", "")),
            image_urls=[str(item) for item in payload.get("image_urls", [])],
            local_image_paths=[str(item) for item in payload.get("local_image_paths", [])],
        )
