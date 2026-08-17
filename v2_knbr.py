from __future__ import annotations

from urllib.parse import urljoin

import feedparser
import requests
from bs4 import BeautifulSoup

from v2_probe import Article, clean, make_article

EXECUTIVE_SHOW_PLAYLIST_URL = "https://omny.fm/shows/the-executives-show/playlists/podcast"
UA = "Mozilla/5.0 GiantsNewsBotV2KNBR/1.0"
TIMEOUT = 15

GIANTS_EXECUTIVE_TERMS = (
    "giants",
    "buster posey",
    "zack minasian",
    "zach minasian",
    "tony vitello",
    "larry baer",
    "greg johnson",
)
NON_GIANTS_EXECUTIVE_TERMS = (
    "49ers",
    "john lynch",
    "al guido",
)


def _rss_url_from_playlist(html: str, page_url: str = EXECUTIVE_SHOW_PLAYLIST_URL) -> str:
    soup = BeautifulSoup(html, "lxml")
    for link in soup.find_all("link", href=True):
        rel = " ".join(link.get("rel") or []).lower()
        media_type = str(link.get("type") or "").lower()
        if "alternate" in rel and "rss" in media_type:
            return urljoin(page_url, link["href"])
    for anchor in soup.find_all("a", href=True):
        text = clean(anchor.get_text(" ")).lower()
        href = str(anchor.get("href") or "")
        if "rss" in text or href.lower().endswith((".rss", "/rss")):
            return urljoin(page_url, href)
    return ""


def _giants_episode(title: str, summary: str) -> bool:
    blob = f"{title} {summary}".lower()
    if any(term in blob for term in NON_GIANTS_EXECUTIVE_TERMS) and not any(
        term in blob for term in GIANTS_EXECUTIVE_TERMS
    ):
        return False
    return any(term in blob for term in GIANTS_EXECUTIVE_TERMS)


def discover_knbr_executive_show() -> list[Article]:
    """Discover the Giants-only Executive Show audio from its Omny playlist RSS."""
    response = requests.get(
        EXECUTIVE_SHOW_PLAYLIST_URL,
        headers={"User-Agent": UA},
        timeout=TIMEOUT,
        allow_redirects=True,
    )
    response.raise_for_status()
    rss_url = _rss_url_from_playlist(response.text, response.url)
    if not rss_url:
        raise RuntimeError("Executive Show playlist did not expose an RSS feed")

    feed = feedparser.parse(rss_url, request_headers={"User-Agent": UA})
    status = getattr(feed, "status", None)
    if status and status >= 400:
        raise RuntimeError(f"Executive Show feed HTTP {status}")
    if getattr(feed, "bozo", False) and not feed.entries:
        raise RuntimeError(f"invalid Executive Show feed: {getattr(feed, 'bozo_exception', '')}")

    out: list[Article] = []
    seen: set[str] = set()
    for entry in feed.entries[:24]:
        title = clean(getattr(entry, "title", ""))
        summary = clean(getattr(entry, "summary", "") or getattr(entry, "description", ""))
        url = str(getattr(entry, "link", "") or "")
        if not title or not url or url in seen or not _giants_episode(title, summary):
            continue
        seen.add(url)
        out.append(make_article(
            source="KNBR",
            title=title,
            url=url,
            published=getattr(entry, "published", "") or getattr(entry, "updated", "") or "",
            author="The Executive Show",
            summary=summary,
            section="The Executive Show · Omny RSS",
            access="free",
        ))
    return out
