from __future__ import annotations

from typing import List
from urllib.parse import urlencode

import feedparser

from config import Settings, SourceConfig
from filters import is_relevant_giants
from models import Candidate


MAX_GOOGLE_CANDIDATES_PER_SOURCE = 5


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
        accepted = 0
        for e in feed.entries[: settings.max_rss_entries_per_feed]:
            src = getattr(e, "source", {})
            src_title = src.get("title", "") if isinstance(src, dict) else ""
            title = getattr(e, "title", "") or ""
            summary = getattr(e, "summary", "") or ""
            published_ts = getattr(e, "published", "") or getattr(e, "updated", "") or ""
            link = getattr(e, "link", "") or ""

            # Google site queries can still surface unrelated stories from the same
            # publication. Filter on the feed metadata before paying the cost to
            # resolve opaque Google News URLs.
            if not is_relevant_giants(title, summary, [], ""):
                continue

            out.append(
                Candidate(
                    source=source.name,
                    discovered_via="google",
                    url=link,
                    feed_url=link,
                    google_url=link,
                    title=title,
                    author=src_title,
                    summary=summary,
                    published_ts=published_ts,
                )
            )
            accepted += 1
            # main.py validates only a bounded candidate set. Keep each Google
            # source small enough that early sources cannot crowd out later ones.
            if accepted >= MAX_GOOGLE_CANDIDATES_PER_SOURCE:
                break
    return out
