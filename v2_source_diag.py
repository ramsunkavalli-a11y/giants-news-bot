from __future__ import annotations

import json
from datetime import datetime, timezone

import feedparser
import requests

UA = "Mozilla/5.0 GiantsNewsBotV2Probe/0.2"
TIMEOUT = 20

HTTP_TARGETS = [
    "https://www.sfchronicle.com/sports/giants/",
    "https://www.sfchronicle.com/sitemap/2026/",
    "https://www.sfgate.com/giants/",
    "https://www.baseballamerica.com/stories/teams/2019-san-francisco-giants/",
]

FEED_TARGETS = [
    "https://www.sfchronicle.com/sports/giants/feed/",
    "https://www.sfgate.com/giants/feed/Giants-447.php",
    "https://www.baseballamerica.com/feed/",
    "https://blogs.fangraphs.com/category/teams/giants/feed/",
]


def main() -> None:
    result = {"generated_at": datetime.now(timezone.utc).isoformat(), "http": [], "feeds": []}
    for url in HTTP_TARGETS:
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
            text = r.text
            result["http"].append({
                "url": url,
                "status": r.status_code,
                "content_type": r.headers.get("content-type", ""),
                "length": len(text),
                "article_links": text.count("/article/"),
                "giants_article_links": text.count("/giants/article/"),
                "title_hint": text[:200].replace("\n", " "),
            })
        except Exception as exc:
            result["http"].append({"url": url, "error": f"{type(exc).__name__}: {exc}"})

    for url in FEED_TARGETS:
        feed = feedparser.parse(url, request_headers={"User-Agent": UA})
        result["feeds"].append({
            "url": url,
            "status": getattr(feed, "status", None),
            "bozo": bool(getattr(feed, "bozo", False)),
            "entries": len(feed.entries),
            "title": getattr(feed.feed, "title", ""),
            "sample": [getattr(e, "title", "") for e in feed.entries[:5]],
        })

    with open("v2-source-diag.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
