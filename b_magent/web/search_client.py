from __future__ import annotations

from typing import Any, Protocol

import httpx

from .models import WebSearchResult


class WebSearchClient(Protocol):
    def search(self, query: str, limit: int = 5) -> list[WebSearchResult]:
        """Search the web and return result metadata without fetching pages."""


class HttpWebSearchClient:
    """Configurable adapter for common JSON web-search APIs."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        *,
        provider: str = "generic",
        client: httpx.Client | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.provider = provider.strip().lower() or "generic"
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=False)

    def search(self, query: str, limit: int = 5) -> list[WebSearchResult]:
        if not query.strip() or limit <= 0:
            return []
        if self.provider == "serper":
            response = self.client.post(
                self.endpoint,
                json={"q": query, "num": limit},
                headers=self._headers(),
            )
        else:
            response = self.client.get(
                self.endpoint,
                params=self._params(query, limit),
                headers=self._headers(),
            )
        response.raise_for_status()
        payload = response.json()
        return self._parse_results(payload, limit)

    def _headers(self) -> dict[str, str]:
        if self.provider == "bing":
            return {"Ocp-Apim-Subscription-Key": self.api_key}
        if self.provider == "brave":
            return {"X-Subscription-Token": self.api_key, "Accept": "application/json"}
        if self.provider == "serper":
            return {"X-API-KEY": self.api_key, "Accept": "application/json"}
        return {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}

    def _params(self, query: str, limit: int) -> dict[str, object]:
        if self.provider == "bing":
            return {"q": query, "count": limit, "responseFilter": "Webpages"}
        if self.provider == "brave":
            return {"q": query, "count": limit}
        if self.provider == "serper":
            return {"q": query, "num": limit}
        return {"q": query, "limit": limit}

    def _parse_results(self, payload: Any, limit: int) -> list[WebSearchResult]:
        items: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            if isinstance(payload.get("webPages"), dict):
                items = _dict_items(payload["webPages"].get("value"))
            elif isinstance(payload.get("web"), dict):
                items = _dict_items(payload["web"].get("results"))
            elif isinstance(payload.get("organic"), list):
                items = _dict_items(payload.get("organic"))
            elif isinstance(payload.get("results"), list):
                items = _dict_items(payload.get("results"))
        results: list[WebSearchResult] = []
        for item in items[:limit]:
            url = str(item.get("url") or item.get("link") or "").strip()
            if not url:
                continue
            results.append(
                WebSearchResult(
                    title=str(item.get("name") or item.get("title") or "").strip(),
                    url=url,
                    snippet=str(item.get("snippet") or item.get("description") or "").strip(),
                    source=self.provider,
                    image_urls=_result_image_urls(item),
                )
            )
        return results


def _dict_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _result_image_urls(item: dict[str, Any]) -> list[str]:
    candidates = [item.get("thumbnailUrl"), item.get("image"), item.get("thumbnail")]
    return [str(value) for value in candidates if isinstance(value, str) and value.startswith("http")]
