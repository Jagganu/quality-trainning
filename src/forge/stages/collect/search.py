"""Web search engine — queries DuckDuckGo for URLs."""

from __future__ import annotations

import asyncio
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup

from forge.utils.logging import get_logger

logger = get_logger(__name__)


class SearchEngine:
    """Queries DuckDuckGo HTML search and extracts result URLs."""

    def __init__(self, max_results: int = 10) -> None:
        self._max_results = max_results

    async def search(self, query: str) -> list[str]:
        """Return up to *max_results* URLs for *query*."""
        urls: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={"User-Agent": "ForgeGravity/0.1"},
                )
                resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            for a_tag in soup.select("a.result__a"):
                href = a_tag.get("href", "")
                if not href:
                    continue
                if href.startswith("http"):
                    urls.append(href)
                elif "uddg=" in href:
                    parsed = urlparse(href)
                    params = parse_qs(parsed.query)
                    if "uddg" in params:
                        urls.append(params["uddg"][0])
                if len(urls) >= self._max_results:
                    break

            logger.info("Search '%s' → %d URLs", query[:50], len(urls))
        except Exception as e:
            logger.warning("Search failed for '%s': %s", query[:50], e)

        await asyncio.sleep(1.0)  # Rate limit
        return urls
