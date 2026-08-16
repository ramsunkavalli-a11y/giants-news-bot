from __future__ import annotations

import json
from datetime import datetime, timezone

import feedparser

from v2_authors import author_prior

UA = "Mozilla/5.0 GiantsNewsBotV2Probe/0.1"

CANDIDATE_FEEDS = [
    "https://www.nytimes.com/athletic/rss/mlb/team/sf-giants/",
    "https://www.nytimes.com/athletic/rss/mlb/teams/sf-giants/",
    "https://www.nytimes.com/athletic/rss/mlb/sf-giants/",
    "https://www.nytimes.com/athletic/rss/mlb/",
    "https://www.nytimes.com/athletic/rss/news/",
]


def clean(value: str) -> str:
    return " ".join((value or "").split())


def main() -> None:
    results = []
    for url in CANDIDATE_FEEDS:
        feed = feedparser.parse(url, request_headers={"User-Agent": UA})
        entries = []
        for entry in feed.entries[:30]:
            title = clean(getattr(entry, "title", ""))
            author = clean(getattr(entry, "author", ""))
            link = getattr(entry, "link", "") or ""
            giants_specific = "giants" in title.lower() or "san francisco" in title.lower()
            prior = author_prior(author)
            entries.append(
                {
                    "title": title,
                    "author": author,
                    "link": link,
                    "giants_specific": giants_specific,
                    "author_prior": prior,
                }
            )
        results.append(
            {
                "url": url,
                "status": getattr(feed, "status", None),
                "href": getattr(feed, "href", ""),
                "bozo": bool(getattr(feed, "bozo", False)),
                "bozo_exception": str(getattr(feed, "bozo_exception", "")) if getattr(feed, "bozo", False) else "",
                "feed_title": clean(getattr(feed.feed, "title", "")),
                "entry_count": len(feed.entries),
                "sample_entries": entries,
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feeds": results,
    }
    with open("v2-athletic-probe.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    for result in results:
        print(
            f"ATHLETIC_FEED status={result['status']} entries={result['entry_count']} "
            f"title={result['feed_title']!r} url={result['url']}"
        )
        for entry in result["sample_entries"][:8]:
            if entry["giants_specific"] or entry["author_prior"]:
                print(
                    f"  candidate author={entry['author']!r} prior={entry['author_prior']} "
                    f"giants={entry['giants_specific']} title={entry['title']}"
                )


if __name__ == "__main__":
    main()
