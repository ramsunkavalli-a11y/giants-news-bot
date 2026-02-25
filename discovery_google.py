from __future__ import annotations

from typing import List
from urllib.parse import urlencode

import feedparser

from config import Settings, SourceConfig
from models import Candidate


def google_news_rss_url(query: str, hours_back: int) -> str:
    q = urlencode({"q": f"{query} when:{hours_back}h", "hl": "en-US", "gl": "US", "ceid": "US:en"})
    return f"https://news.google.com/rss/search?{q}"


def discover_google_news(settings: Settings, sources: List[SourceConfig]) -> List[Candidate]:
    out: List[Candidate] = []
    for source in sources:
        feed = feedparser.parse(
            google_news_rss_url(source.google_query, settings.hours_back),
            request_headers={"User-Agent": settings.user_agent},
        )
        for e in feed.entries[: settings.max_rss_entries_per_feed]:
            src = getattr(e, "source", {})
            src_title = src.get("title", "") if isinstance(src, dict) else ""
            out.append(
                Candidate(
                    source=source.name,
                    discovered_via="google",
                    url=getattr(e, "link", "") or "",
                    title=getattr(e, "title", "") or "",
                    author=src_title,
                    summary=getattr(e, "summary", "") or "",
                    published_ts=getattr(e, "published", "") or getattr(e, "updated", "") or "",
                )
            )
    return out
