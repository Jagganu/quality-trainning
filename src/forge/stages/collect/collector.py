"""Collect stage — generates search queries, scrapes pages, deduplicates content."""

from __future__ import annotations

import json
from urllib.parse import urlparse

from forge.core.context import PipelineContext
from forge.core.stage import Stage
from forge.providers.llm import LLMProvider
from forge.stages.collect.dedupe import DeduplicationEngine
from forge.stages.collect.scraper import WebScraper
from forge.stages.collect.search import SearchEngine
from forge.utils.logging import get_logger

logger = get_logger(__name__)


def _is_blocked_domain(url: str, blocked: list[str]) -> bool:
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return False
    host = host.lower().removeprefix("www.")
    return any(host == b or host.endswith(f".{b}") for b in blocked)


class CollectStage(Stage):
    """Stage 1: Collect raw documents from the web."""

    @property
    def name(self) -> str:
        return "collect"

    async def run(self, context: PipelineContext) -> PipelineContext:
        settings = context.settings.collect
        llm = LLMProvider(context.settings, context.budget)

        # 1. Generate search queries via LLM
        logger.info("Generating search queries for '%s'…", context.topic)
        raw = await llm.complete(
            prompt=(
                f"Generate 10 diverse search queries for collecting comprehensive training data "
                f"about: {context.topic}. Cover fundamentals, advanced topics, practical examples, "
                f"and edge cases. Return ONLY a JSON array of strings."
            ),
            system="You are a research assistant. Return only valid JSON.",
            stage="collect",
        )
        try:
            queries = json.loads(raw.text)
            if not isinstance(queries, list):
                queries = [context.topic]
        except json.JSONDecodeError:
            queries = [
                context.topic,
                f"{context.topic} tutorial",
                f"{context.topic} advanced guide",
            ]

        logger.info("Generated %d search queries", len(queries))

        # 2. Search for URLs
        search = SearchEngine(max_results=8)
        all_urls: list[str] = []
        seen_urls: set[str] = set()
        for query in queries:
            urls = await search.search(query)
            for url in urls:
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_urls.append(url)

        # 3. Filter blocked domains before scraping
        blocked = settings.blocked_domains
        filtered_urls = [u for u in all_urls if not _is_blocked_domain(u, blocked)]
        blocked_count = len(all_urls) - len(filtered_urls)
        if blocked_count:
            logger.info("Blocked %d URLs from low-quality domains", blocked_count)

        filtered_urls = filtered_urls[: settings.max_pages]
        logger.info(
            "Found %d URLs to scrape (capped to %d)",
            len(filtered_urls), settings.max_pages,
        )

        # 4. Scrape pages
        async with WebScraper(settings) as scraper:
            documents = await scraper.scrape_batch(filtered_urls)

        # 5. Filter by word count
        min_wc = settings.min_word_count
        before = len(documents)
        documents = [d for d in documents if d.word_count >= min_wc]
        if before != len(documents):
            logger.info("Dropped %d docs with < %d words", before - len(documents), min_wc)

        context.metrics.increment("pages_scraped", len(filtered_urls))
        logger.info("Scraped %d pages → %d documents", len(filtered_urls), len(documents))

        # 6. Deduplicate content
        dedupe = DeduplicationEngine()
        unique_docs = []
        for doc in documents:
            result = dedupe.add_document(doc.doc_id, doc.content)
            if not result.is_duplicate:
                unique_docs.append(doc)

        context.documents = unique_docs
        context.dedup_report = dedupe.report()
        context.metrics.increment("documents_collected", len(unique_docs))
        logger.info(
            "After dedup: %d unique docs (%d exact dupes, %d near dupes)",
            context.dedup_report.unique_documents,
            context.dedup_report.exact_duplicates,
            context.dedup_report.near_duplicates,
        )

        # 7. Persist documents
        for doc in unique_docs:
            await context.storage.save_document("collected", doc.doc_id, doc.model_dump())

        return context
