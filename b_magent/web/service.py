from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from .models import WebSearchResult, utc_now
from .image_fetcher import ImageFetcher
from .page_fetcher import PageFetcher
from .search_client import HttpWebSearchClient, WebSearchClient


class WebSearchCache(Protocol):
    def lookup(self, queries: list[str]) -> list[WebSearchResult]: ...

    def store(
        self,
        question: str,
        queries: list[str],
        results: list[WebSearchResult],
        ttl_seconds: int,
    ) -> None: ...


class WebSearchService:
    def __init__(
        self,
        search_client: WebSearchClient,
        page_fetcher: PageFetcher,
        cache: WebSearchCache,
        image_fetcher: ImageFetcher | None = None,
        *,
        max_queries: int = 4,
        results_per_query: int = 5,
        max_pages: int = 8,
        max_images_per_page: int = 2,
        max_images: int = 8,
    ) -> None:
        self.search_client = search_client
        self.page_fetcher = page_fetcher
        self.cache = cache
        self.image_fetcher = image_fetcher
        self.max_queries = max_queries
        self.results_per_query = results_per_query
        self.max_pages = max_pages
        self.max_images_per_page = max_images_per_page
        self.max_images = max_images

    def search_and_fetch(
        self,
        question: str,
        queries: list[str],
        *,
        ttl_seconds: int = 86_400,
    ) -> tuple[list[WebSearchResult], bool]:
        normalized_queries = list(
            dict.fromkeys(" ".join(query.split()) for query in queries if query.strip())
        )[: self.max_queries]
        if not normalized_queries:
            return [], False
        cached = self.cache.lookup(normalized_queries)
        if cached:
            return cached, True

        candidates: list[WebSearchResult] = []
        seen_urls: set[str] = set()
        for query in normalized_queries:
            try:
                query_results = self.search_client.search(query, limit=self.results_per_query)
            except Exception:
                continue
            for result in query_results:
                normalized_url = result.url.strip().rstrip("/")
                if not normalized_url or normalized_url in seen_urls:
                    continue
                seen_urls.add(normalized_url)
                candidates.append(result)

        enriched: list[WebSearchResult] = []
        downloaded_image_count = 0
        for result in candidates[: self.max_pages]:
            image_urls = list(result.image_urls)
            try:
                fetch_page = getattr(self.page_fetcher, "fetch_page", None)
                if callable(fetch_page):
                    page = fetch_page(result.url)
                    content = page.text
                    image_urls.extend(page.image_urls)
                else:
                    content = self.page_fetcher.fetch(result.url)
            except Exception:
                content = ""
            image_urls = list(dict.fromkeys(image_urls))
            local_image_paths: list[str] = []
            if self.image_fetcher is not None and downloaded_image_count < self.max_images:
                for image_url in image_urls[: self.max_images_per_page]:
                    if downloaded_image_count >= self.max_images:
                        break
                    try:
                        image_path = self.image_fetcher.fetch(image_url)
                    except Exception:
                        continue
                    local_image_paths.append(str(image_path.resolve()))
                    downloaded_image_count += 1
            enriched.append(
                WebSearchResult(
                    title=result.title,
                    url=result.url,
                    snippet=result.snippet,
                    content=content,
                    source=result.source,
                    retrieved_at=utc_now(),
                    image_urls=image_urls,
                    local_image_paths=local_image_paths,
                )
            )
        if enriched:
            self.cache.store(question, normalized_queries, enriched, ttl_seconds)
        return enriched, False


def build_web_search_service_from_env(data_dir: Path) -> WebSearchService | None:
    endpoint = os.environ.get("WEB_SEARCH_ENDPOINT", "").strip()
    api_key = os.environ.get("WEB_SEARCH_API_KEY", "").strip()
    if not endpoint or not api_key:
        return None
    provider = os.environ.get("WEB_SEARCH_PROVIDER", "generic").strip()
    from s_server.web_search_store import ServerWebSearchStore

    return WebSearchService(
        HttpWebSearchClient(endpoint, api_key, provider=provider),
        PageFetcher(),
        ServerWebSearchStore(data_dir / "qwen_server_agent" / "web_search_store.jsonl"),
        ImageFetcher(data_dir / "qwen_server_agent" / "web_images"),
    )
