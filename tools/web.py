"""
tools/web.py — Web interaction tools for KOBRA.

Functions:
  open_url    — open a URL in the default browser
  web_search  — scrape DuckDuckGo and return top result snippets
"""

import logging
import webbrowser
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
_MAX_SNIPPET_CHARS = 600
_DDG_HTML_URL = "https://html.duckduckgo.com/html/?q={query}"


def open_url(url: str) -> str:
    """Open a URL in the user's default browser."""
    logger.info("[TOOL] open_url: %s", url)
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    webbrowser.open(url)
    return f"Opened {url}."


def web_search(query: str) -> str:
    """
    Search DuckDuckGo and return the top 3 result snippets concatenated.
    Falls back to opening the browser if the HTTP request fails.
    """
    logger.info("[TOOL] web_search: %r", query)
    url = _DDG_HTML_URL.format(query=quote_plus(query))

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("web_search HTTP failed: %s — falling back to browser.", exc)
        fallback_url = f"https://duckduckgo.com/?q={quote_plus(query)}"
        webbrowser.open(fallback_url)
        return "I couldn't retrieve search results directly. I've opened the browser for you."

    soup = BeautifulSoup(resp.text, "html.parser")

    snippets: list[str] = []
    for result in soup.select(".result__body"):
        snippet_tag = result.select_one(".result__snippet")
        title_tag = result.select_one(".result__title")
        if snippet_tag:
            title = title_tag.get_text(strip=True) if title_tag else ""
            snippet = snippet_tag.get_text(strip=True)
            snippets.append(f"{title}: {snippet}" if title else snippet)
        if len(snippets) >= 3:
            break

    if not snippets:
        # Fallback: open browser
        fallback_url = f"https://duckduckgo.com/?q={quote_plus(query)}"
        webbrowser.open(fallback_url)
        return "No results scraped. I've opened a browser search for you."

    combined = " | ".join(snippets)
    if len(combined) > _MAX_SNIPPET_CHARS:
        combined = combined[:_MAX_SNIPPET_CHARS - 1] + "…"
    return combined
