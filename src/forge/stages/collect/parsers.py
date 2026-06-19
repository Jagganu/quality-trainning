"""Document parsers — HTML, PDF, content-type detection."""

from __future__ import annotations

from bs4 import BeautifulSoup


def parse_html(html: str) -> str:
    """Extract plain text from HTML, stripping all tags."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def parse_pdf(content: bytes) -> str:
    """Extract text from PDF bytes (stub for MVP)."""
    return "[PDF parsing not yet implemented — install pdfplumber for PDF support]"


def detect_content_type(url: str, headers: dict[str, str] | None = None) -> str:
    """Detect content type from HTTP headers or URL extension."""
    if headers:
        ct = headers.get("content-type", "").lower()
        if "pdf" in ct:
            return "application/pdf"
        if "html" in ct:
            return "text/html"
        if "json" in ct:
            return "application/json"
        if "text" in ct:
            return "text/plain"

    url_lower = url.lower()
    if url_lower.endswith(".pdf"):
        return "application/pdf"
    if url_lower.endswith((".md", ".txt")):
        return "text/plain"
    return "text/html"
