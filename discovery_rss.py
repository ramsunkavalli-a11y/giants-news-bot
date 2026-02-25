from __future__ import annotations

from typing import Dict, List, Tuple

import feedparser

from config import Settings, SourceConfig
from models import Candidate


def discover_rss_sources(settings: Settings, sources: List[SourceConfig]) -> Tuple[List[Candidate], Dict[str, dict]]:
    candidates: List[Candidate] = []
    health: Dict[str, dict] = {}
    for source in sources:
        feed = feedparser.parse(source.rss_url, request_headers={"User-Agent": settings.user_agent})
        entries = feed.entries[: settings.max_rss_entries_per_feed]
        health[source.name] = {
            "feed_url": source.rss_url,
            "bozo": bool(getattr(feed, "bozo", False)),
            "entry_count": len(entries),
            "status": getattr(feed, "status", None),
            "bozo_exception": str(getattr(feed, "bozo_exception", "")) if getattr(feed, "bozo", False) else "",
        }
        for e in entries:
            candidates.append(
                Candidate(
                    source=source.name,
                    discovered_via="rss",
                    url=getattr(e, "link", "") or "",
                    feed_url=getattr(e, "link", "") or "",
                    title=getattr(e, "title", "") or "",
                    author=getattr(e, "author", "") or "",
                    summary=getattr(e, "summary", "") or "",
                    categories=[t.get("term", "") for t in getattr(e, "tags", []) if isinstance(t, dict)],
                    published_ts=getattr(e, "published", "") or getattr(e, "updated", "") or getattr(e, "pubDate", "") or "",
                )
            )
    return candidates, health
