from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup


@dataclass
class FetchedPage:
    text: str
    image_urls: list[str]


class PageFetcher:
    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        timeout: float = 15.0,
        max_response_bytes: int = 2_000_000,
        max_content_chars: int = 20_000,
    ) -> None:
        self.client = client or httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            headers={"User-Agent": "BMagentResearchBot/1.0"},
        )
        self.max_response_bytes = max_response_bytes
        self.max_content_chars = max_content_chars

    def fetch(self, url: str) -> str:
        return self.fetch_page(url).text

    def fetch_page(self, url: str) -> FetchedPage:
        current_url = url
        raw = b""
        content_type = ""
        for redirect_count in range(4):
            _validate_public_url(current_url)
            with self.client.stream("GET", current_url, follow_redirects=False) as response:
                if response.is_redirect:
                    location = response.headers.get("location", "")
                    if not location or redirect_count >= 3:
                        raise ValueError(f"too many or invalid redirects for URL: {url!r}")
                    current_url = urljoin(current_url, location)
                    continue
                response.raise_for_status()
                _validate_public_url(str(response.url))
                content_type = response.headers.get("content-type", "").lower()
                if "text/html" not in content_type and "text/plain" not in content_type:
                    return FetchedPage(text="", image_urls=[])
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    remaining = self.max_response_bytes - total
                    if remaining <= 0:
                        break
                    chunks.append(chunk[:remaining])
                    total += min(len(chunk), remaining)
                raw = b"".join(chunks)
                break
        soup = BeautifulSoup(raw, "html.parser")
        image_urls = _extract_image_urls(soup, current_url)
        for element in soup(["script", "style", "nav", "footer", "noscript", "svg"]):
            element.decompose()
        text = " ".join(soup.get_text(" ").split())
        return FetchedPage(text=text[: self.max_content_chars], image_urls=image_urls)


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"unsupported web URL: {url!r}")
    host = parsed.hostname.lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError(f"local web URL is not allowed: {url!r}")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port or 443)}
    except socket.gaierror as exc:
        raise ValueError(f"cannot resolve web URL host: {host}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError(f"non-public web URL is not allowed: {url!r}")


def _extract_image_urls(soup: BeautifulSoup, page_url: str, limit: int = 8) -> list[str]:
    candidates: list[str] = []
    for meta in soup.select('meta[property="og:image"], meta[name="twitter:image"]'):
        candidates.append(str(meta.get("content", "")))
    for image in soup.find_all("img"):
        source = str(image.get("src") or image.get("data-src") or "")
        if not source and image.get("srcset"):
            source = str(image.get("srcset")).split(",")[-1].strip().split(" ")[0]
        candidates.append(source)
    results: list[str] = []
    for candidate in candidates:
        if not candidate or candidate.startswith("data:"):
            continue
        resolved = urljoin(page_url, candidate)
        if resolved not in results:
            results.append(resolved)
        if len(results) >= limit:
            break
    return results
