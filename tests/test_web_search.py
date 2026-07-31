from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import httpx
from PIL import Image

from _project_path import add_project_root_to_sys_path

add_project_root_to_sys_path()

from b_magent.web import (
    HttpWebSearchClient,
    ImageFetcher,
    PageFetcher,
    WebSearchResult,
    WebSearchService,
)
from s_server import ServerWebSearchStore


class FakeSearchClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str, limit: int = 5) -> list[WebSearchResult]:
        self.calls.append(query)
        return [
            WebSearchResult(
                title=f"Result for {query}",
                url="https://example.com/facts",
                snippet="current verified facts",
                source="fake",
            )
        ]


class FakePageFetcher:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def fetch(self, url: str) -> str:
        self.calls.append(url)
        return "Official current price is 25 dollars."


class WebSearchTestCase(unittest.TestCase):
    def test_page_fetcher_discovers_open_graph_and_body_images(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=(
                    '<html><head><meta property="og:image" content="/cover.jpg"></head>'
                    '<body><main>Product details<img src="https://cdn.example.com/item.png"></main></body></html>'
                ),
                request=request,
            )

        fetcher = PageFetcher(client=httpx.Client(transport=httpx.MockTransport(handler)))
        public_address = [(None, None, None, None, ("93.184.216.34", 443))]

        with patch("socket.getaddrinfo", return_value=public_address):
            page = fetcher.fetch_page("https://example.com/product")

        self.assertEqual(page.text, "Product details")
        self.assertEqual(
            page.image_urls,
            ["https://example.com/cover.jpg", "https://cdn.example.com/item.png"],
        )

    def test_image_fetcher_validates_and_persists_image(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_image_fetch_test_"))
        try:
            buffer = BytesIO()
            Image.new("RGB", (64, 64), color="red").save(buffer, format="PNG")

            def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200,
                    headers={"content-type": "image/png"},
                    content=buffer.getvalue(),
                    request=request,
                )

            fetcher = ImageFetcher(
                temp_dir,
                client=httpx.Client(transport=httpx.MockTransport(handler)),
            )
            public_address = [(None, None, None, None, ("93.184.216.34", 443))]

            with patch("socket.getaddrinfo", return_value=public_address):
                image_path = fetcher.fetch("https://example.com/image.png")

            self.assertTrue(image_path.is_file())
            with Image.open(image_path) as image:
                self.assertEqual(image.size, (64, 64))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_http_search_client_parses_bing_json_results(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.params["q"], "current price")
            self.assertEqual(request.headers["Ocp-Apim-Subscription-Key"], "secret")
            return httpx.Response(
                200,
                json={
                    "webPages": {
                        "value": [
                            {
                                "name": "Official price",
                                "url": "https://example.com/price",
                                "snippet": "The current price is listed here.",
                            }
                        ]
                    }
                },
            )

        client = HttpWebSearchClient(
            "https://search.example.test/v1",
            "secret",
            provider="bing",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        results = client.search("current price", limit=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Official price")
        self.assertEqual(results[0].url, "https://example.com/price")

    def test_page_fetcher_rejects_local_and_private_urls(self) -> None:
        fetcher = PageFetcher()

        with self.assertRaises(ValueError):
            fetcher.fetch("http://localhost/admin")
        with self.assertRaises(ValueError):
            fetcher.fetch("http://127.0.0.1/private")

    def test_web_search_service_uses_ttl_cache_and_deduplicates_urls(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="b_magent_web_cache_test_"))
        try:
            search_client = FakeSearchClient()
            page_fetcher = FakePageFetcher()
            store = ServerWebSearchStore(temp_dir / "web_search_store.jsonl")
            service = WebSearchService(search_client, page_fetcher, store)

            first, first_cache_hit = service.search_and_fetch(
                "What is the current price?",
                ["current official price", "current official price"],
                ttl_seconds=3600,
            )
            second, second_cache_hit = service.search_and_fetch(
                "What is the current price?",
                ["current official price"],
                ttl_seconds=3600,
            )

            self.assertFalse(first_cache_hit)
            self.assertTrue(second_cache_hit)
            self.assertEqual(len(first), 1)
            self.assertEqual(first, second)
            self.assertEqual(search_client.calls, ["current official price"])
            self.assertEqual(page_fetcher.calls, ["https://example.com/facts"])
            payload = json.loads(store.path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("expires_at", payload)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
