"""Web scraper — async HTTP fetching + trafilatura content extraction."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import trafilatura

from forge.core.config import CollectSettings
from forge.core.models import Document
from forge.utils.hashing import md5_hash
from forge.utils.logging import get_logger

logger = get_logger(__name__)


class WebScraper:
    """Fetches web pages and extracts main content."""

    def __init__(self, settings: CollectSettings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(int(settings.requests_per_second * 2) or 4)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=20.0,
                follow_redirects=True,
                headers={"User-Agent": self._settings.user_agent},
            )
        return self._client

    async def scrape(self, url: str) -> Document | None:
        """Fetch and extract content from a single URL."""
        min_len = self._settings.min_content_length
        try:
            client = await self._get_client()
            async with self._semaphore:
                resp = await client.get(url)
                resp.raise_for_status()
                await asyncio.sleep(1.0 / self._settings.requests_per_second)

            content_type = resp.headers.get("content-type", "")
            if "pdf" in content_type:
                logger.debug("Skipping %s (PDF, not HTML)", url)
                return None

            html = resp.text
            content = trafilatura.extract(html, include_comments=False, include_tables=True) or ""
            if not content or len(content) < min_len:
                logger.debug("Skipping %s (extractable content < %d chars)", url, min_len)
                return None

            title = ""
            meta = trafilatura.extract(html, output_format="json", include_comments=False)
            if meta:
                import json
                try:
                    parsed = json.loads(meta)
                    title = parsed.get("title", "")
                except Exception:
                    pass

            doc = Document(
                url=url,
                title=title,
                content=content,
                content_hash=md5_hash(content),
                word_count=len(content.split()),
            )
            logger.debug("Scraped %s (%d words)", url, doc.word_count)
            return doc

        except Exception as e:
            logger.warning("Scrape failed %s: %s", url, e)
            return None

    async def scrape_batch(self, urls: list[str]) -> list[Document]:
        """Scrape multiple URLs concurrently."""
        results: list[Document] = []
        tasks = [self.scrape(url) for url in urls]
        for coro in asyncio.as_completed(tasks):
            doc = await coro
            if doc is not None:
                results.append(doc)
        return results

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> WebScraper:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
