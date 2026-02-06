"""
Web tools: web_search and web_fetch.

ClawAI Agent style:
- Declarative tool metadata
- Explicit validation & safety boundaries
- Structured JSON outputs
"""

from __future__ import annotations

import html
import json
import os
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from clawai.tools.base import Tool


# =============================================================================
# Constants
# =============================================================================

USER_AGENT = ""
DEFAULT_TIMEOUT = 30.0
MAX_REDIRECTS = 5
MAX_SEARCH_RESULTS = 10


# =============================================================================
# Text utilities
# =============================================================================

_SCRIPT_RE = re.compile(r"<script[\s\S]*?</script>", re.I)
_STYLE_RE = re.compile(r"<style[\s\S]*?</style>", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = _SCRIPT_RE.sub("", text)
    text = _STYLE_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    return html.unescape(text).strip()


def normalize_text(text: str) -> str:
    """Normalize whitespace and newlines."""
    text = _WS_RE.sub(" ", text)
    return _NL_RE.sub("\n\n", text).strip()


# =============================================================================
# URL validation
# =============================================================================

def validate_url(url: str) -> tuple[bool, str]:
    """Validate URL format and scheme."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False, "Only http/https URLs are allowed"
        if not parsed.netloc:
            return False, "URL missing domain"
        return True, ""
    except Exception as exc:
        return False, str(exc)


# =============================================================================
# Web Search Tool
# =============================================================================

class WebSearchTool(Tool):
    """
    Search the web using Brave Search API.
    """

    name = "web_search"
    description = "Search the web and return a list of results (title, url, snippet)."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "count": {
                "type": "integer",
                "description": "Number of results (1–10)",
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
    }

    def __init__(self, api_key: str | None = None, default_count: int = 5):
        self._api_key = api_key or os.getenv("BRAVE_API_KEY", "")
        self._default_count = min(max(default_count, 1), MAX_SEARCH_RESULTS)

    async def execute(
        self,
        query: str,
        count: int | None = None,
        **_: Any,
    ) -> str:
        if not self._api_key:
            return json.dumps({"error": "BRAVE_API_KEY not configured"})

        n = min(max(count or self._default_count, 1), MAX_SEARCH_RESULTS)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "",
                    params={"q": query, "count": n},
                    headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": self._api_key,
                        "User-Agent": USER_AGENT,
                    },
                )
                resp.raise_for_status()

            results = resp.json().get("web", {}).get("results", [])
            items = [
                {
                    "rank": i + 1,
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("description", ""),
                }
                for i, r in enumerate(results[:n])
            ]

            return json.dumps(
                {
                    "query": query,
                    "count": len(items),
                    "results": items,
                },
                ensure_ascii=False,
            )

        except Exception as exc:
            return json.dumps({"error": str(exc), "query": query})


# =============================================================================
# Web Fetch Tool
# =============================================================================

class WebFetchTool(Tool):
    """
    Fetch a URL and extract readable content.

    - HTML → Readability → markdown / text
    - JSON passthrough
    """

    name = "web_fetch"
    description = "Fetch a URL and extract readable content."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch",
            },
            "extractMode": {
                "type": "string",
                "enum": ["markdown", "text"],
                "default": "markdown",
            },
            "maxChars": {
                "type": "integer",
                "minimum": 100,
                "description": "Maximum characters to return",
            },
        },
        "required": ["url"],
    }

    def __init__(self, default_max_chars: int = 50_000):
        self._default_max_chars = default_max_chars

    async def execute(
        self,
        url: str,
        extractMode: str = "markdown",
        maxChars: int | None = None,
        **_: Any,
    ) -> str:
        from readability import Document

        max_chars = maxChars or self._default_max_chars

        valid, error = validate_url(url)
        if not valid:
            return json.dumps({"error": error, "url": url})

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=MAX_REDIRECTS,
                timeout=DEFAULT_TIMEOUT,
                headers={"User-Agent": USER_AGENT},
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            extractor = "raw"

            # JSON response
            if "application/json" in content_type:
                text = json.dumps(resp.json(), indent=2, ensure_ascii=False)
                extractor = "json"

            # HTML response
            elif "text/html" in content_type or resp.text.lstrip().lower().startswith("<!doctype"):
                doc = Document(resp.text)
                body = doc.summary()

                if extractMode == "markdown":
                    body = self._html_to_markdown(body)
                else:
                    body = strip_html(body)

                title = doc.title()
                text = f"# {title}\n\n{body}" if title else body
                extractor = "readability"

            # Plain text / others
            else:
                text = resp.text

            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]

            return json.dumps(
                {
                    "url": url,
                    "finalUrl": str(resp.url),
                    "status": resp.status_code,
                    "extractor": extractor,
                    "length": len(text),
                    "truncated": truncated,
                    "text": text,
                },
                ensure_ascii=False,
            )

        except Exception as exc:
            return json.dumps({"error": str(exc), "url": url})

    # -------------------------------------------------------------------------

    def _html_to_markdown(self, html_text: str) -> str:
        """Best-effort HTML → markdown conversion."""
        text = re.sub(
            r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
            lambda m: f"[{strip_html(m[2])}]({m[1]})",
            html_text,
            flags=re.I | re.S,
        )
        text = re.sub(
            r"<h([1-6])[^>]*>(.*?)</h\1>",
            lambda m: f"\n{'#' * int(m[1])} {strip_html(m[2])}\n",
            text,
            flags=re.I | re.S,
        )
        text = re.sub(
            r"<li[^>]*>(.*?)</li>",
            lambda m: f"\n- {strip_html(m[1])}",
            text,
            flags=re.I | re.S,
        )
        text = re.sub(r"</(p|div|section|article)>", "\n\n", text, flags=re.I)
        text = re.sub(r"<(br|hr)\s*/?>", "\n", text, flags=re.I)
        return normalize_text(strip_html(text))
