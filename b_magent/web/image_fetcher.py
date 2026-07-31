from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin

import httpx
from PIL import Image

from .page_fetcher import _validate_public_url


class ImageFetcher:
    def __init__(
        self,
        cache_dir: Path,
        *,
        client: httpx.Client | None = None,
        timeout: float = 15.0,
        max_response_bytes: int = 10_000_000,
        max_pixels: int = 40_000_000,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = client or httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            headers={"User-Agent": "BMagentResearchBot/1.0"},
        )
        self.max_response_bytes = max_response_bytes
        self.max_pixels = max_pixels

    def fetch(self, url: str) -> Path:
        cache_key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        existing = next(self.cache_dir.glob(f"{cache_key}.*"), None)
        if existing is not None:
            return existing

        current_url = url
        raw = b""
        for redirect_count in range(4):
            _validate_public_url(current_url)
            with self.client.stream("GET", current_url, follow_redirects=False) as response:
                if response.is_redirect:
                    location = response.headers.get("location", "")
                    if not location or redirect_count >= 3:
                        raise ValueError(f"too many or invalid image redirects: {url!r}")
                    current_url = urljoin(current_url, location)
                    continue
                response.raise_for_status()
                _validate_public_url(str(response.url))
                if not response.headers.get("content-type", "").lower().startswith("image/"):
                    raise ValueError(f"URL did not return an image: {url!r}")
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > self.max_response_bytes:
                        raise ValueError(f"image exceeds size limit: {url!r}")
                    chunks.append(chunk)
                raw = b"".join(chunks)
                break

        with Image.open(BytesIO(raw)) as image:
            image.verify()
        with Image.open(BytesIO(raw)) as image:
            if image.width * image.height > self.max_pixels:
                raise ValueError(f"image exceeds pixel limit: {url!r}")
            image_format = (image.format or "PNG").lower()
        extension = {"jpeg": "jpg", "png": "png", "webp": "webp", "gif": "gif"}.get(
            image_format,
            "img",
        )
        output_path = self.cache_dir / f"{cache_key}.{extension}"
        output_path.write_bytes(raw)
        return output_path
