"""Browser trajectory collector — captures browser interactions.

This module provides a stub implementation that can be extended with
Playwright or Selenium for actual browser automation. Without those
dependencies, it returns placeholder observations.
"""

from __future__ import annotations

from forge.utils.logging import get_logger

logger = get_logger(__name__)


class BrowserTrajectoryCollector:
    """Collects browser interaction trajectories."""

    def __init__(self) -> None:
        self._page_url: str = ""
        self._page_content: str = ""

    async def observe_page(self) -> str:
        """Capture current page state as an observation string.

        In production, this would use Playwright to capture the DOM,
        screenshot, or accessibility tree. The stub returns a placeholder.
        """
        try:
            # Attempt Playwright if available
            from playwright.async_api import async_playwright  # type: ignore[import-untyped]

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                if self._page_url:
                    await page.goto(self._page_url)
                content = await page.content()
                await browser.close()
                return content[:5000]
        except ImportError:
            logger.debug("Playwright not available — using stub observation")
            return f"[Page observation stub] URL: {self._page_url or 'none'}"

    async def record_action(self, action: str, selector: str | None) -> str:
        """Record a browser action and return its result.

        Parameters
        ----------
        action:
            The action type (e.g. "click", "type", "navigate").
        selector:
            CSS selector for the target element, if applicable.
        """
        result = f"Executed '{action}' on '{selector or 'page'}'"
        logger.debug("Browser action: %s", result)
        return result
