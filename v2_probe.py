from __future__ import annotations

import gzip
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup

UA = "Mozilla/5.0 GiantsNewsBotV2Probe/0.1"
TIMEOUT = 20


@dataclass
class Article:
    source: str
    title: str
    url: str
    published: str = ""
    author: str = ""
    summary: str = ""
    section: str = ""
    quality: str = "high"
    quality_reason: str = "structured_source"


LOW_VALUE_PATTERNS = (
    "press conference",
    "highlights:",
    "highlights ",
    "postgame live",
    "pregame live",
    "stream games",
    "watch live",
    "how to watch",
)

GENERIC_PATTERNS = (
    "power rankings",
    "each team's",
    "every team",
    "for every team",
)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def giants_relevant(text: str) -> bool:
    t = clean(text).lower()
    return "giants" in t or "san francisco" in t


def classify(title: str, section: str = "") -> tuple[str, str]:
    blob = f"{title} {section}".lower()
    if any(p in blob for p in LOW_VALUE_PATTERNS):
        return "low", "commodity_video_or_recap"
    if any(p in blob for p in GENERIC_PATTERNS):
        return "low", "generic_multi_team_content"
    if "what we learned" in blob or "observations" in blob:
        return "medium", "postgame_analysis"
    return "high", "original_news_or_analysis_candidate"


def get(url: str) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
    r.raise_for_status()
    return r


def discover_sf_standard() -> list[Article]:
    feed_url = "https://sfstandard.com/category/sports/feed/"
    feed = feedparser.parse(feed_url, request_headers={"User-Agent": UA})
    out: list[Article] = []
    for e in feed.entries[:30]:
        title = clean(getattr(e, "title", ""))
        summary = clean(getattr(e, "summary", ""))
        if not giants_relevant(f"{title} {summary}"):
            continue
        q, reason = classify(title)
        out.append(Article(
            source="San Francisco Standard",
            title=title,
            url=getattr(e, "link", "") or "",
            published=getattr(e, "published", "") or getattr(e, "updated", "") or "",
            author=clean(getattr(e, "author", "")),
            summary=summary,
            section="Sports RSS",
            quality=q,
            quality_reason=reason,
        ))
    return out


def discover_mlb() -> list[Article]:
    url = "https://www.mlb.com/sitemaps/48-hr-news.xml.gz"
    raw = get(url).content
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    root = ET.fromstring(raw)
    ns = {
        "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
        "news": "http://www.google.com/schemas/sitemap-news/0.9",
    }
    out: list[Article] = []
    for node in root.findall("sm:url", ns):
        loc = clean(node.findtext("sm:loc", default="", namespaces=ns))
        title = clean(node.findtext("news:news/news:title", default="", namespaces=ns))
        published = clean(node.findtext("news:news/news:publication_date", default="", namespaces=ns))
        if not loc or not title or not giants_relevant(title):
            continue
        q, reason = classify(title)
        out.append(Article(
            source="MLB.com",
            title=title,
            url=loc,
            published=published,
            section="48-hour news sitemap",
            quality=q,
            quality_reason=reason,
        ))
    return out


def discover_nbc() -> list[Article]:
    pages = [
        ("Giants News", "https://www.nbcsportsbayarea.com/mlb/san-francisco-giants/giants-news/"),
        ("Giants Analysis", "https://www.nbcsportsbayarea.com/mlb/san-francisco-giants/giants-analysis/"),
    ]
    seen: set[str] = set()
    out: list[Article] = []
    for section, page_url in pages:
        soup = BeautifulSoup(get(page_url).text, "lxml")
        for a in soup.find_all("a", href=True):
            href = urljoin(page_url, a["href"])
            path = urlparse(href).path.rstrip("/")
            title = clean(a.get_text(" "))
            parts = [p for p in path.split("/") if p]
            if len(parts) < 4 or parts[:2] != ["mlb", "san-francisco-giants"]:
                continue
            if not parts[-1].isdigit() or "video" in parts:
                continue
            if not title or len(title) < 20 or href in seen:
                continue
            seen.add(href)
            q, reason = classify(title, section)
            out.append(Article(
                source="NBC Sports Bay Area",
                title=title,
                url=href,
                section=section,
                quality=q,
                quality_reason=reason,
            ))
    return out


def discover_chronicle() -> list[Article]:
    out: list[Article] = []
    seen: set[str] = set()
    now = datetime.now(timezone.utc)
    for days_back in range(3):
        day = now - timedelta(days=days_back)
        month = day.strftime("%B").lower()
        page_url = f"https://www.sfchronicle.com/sitemap/{day.year}/{month}/{day.day}/"
        try:
            soup = BeautifulSoup(get(page_url).text, "lxml")
        except requests.RequestException:
            continue
        for a in soup.find_all("a", href=True):
            href = urljoin(page_url, a["href"])
            title = clean(a.get_text(" "))
            if "/sports/giants/article/" not in href or not title or href in seen:
                continue
            seen.add(href)
            q, reason = classify(title)
            out.append(Article(
                source="San Francisco Chronicle",
                title=title,
                url=href,
                published=day.date().isoformat(),
                section="daily sitemap",
                quality=q,
                quality_reason=reason,
            ))
    return out


def main() -> None:
    discoverers = [discover_sf_standard, discover_mlb, discover_nbc, discover_chronicle]
    all_articles: list[Article] = []
    health = {}
    for fn in discoverers:
        name = fn.__name__.replace("discover_", "")
        try:
            items = fn()
            health[name] = {"ok": True, "count": len(items)}
            all_articles.extend(items)
        except Exception as exc:
            health[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    deduped: dict[str, Article] = {}
    for article in all_articles:
        deduped[article.url] = article

    articles = list(deduped.values())
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "health": health,
        "counts_by_quality": {
            q: sum(1 for a in articles if a.quality == q)
            for q in ("high", "medium", "low")
        },
        "articles": [asdict(a) for a in articles],
    }
    with open("v2-probe.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(payload["health"], indent=2))
    for a in articles:
        print(f"[{a.quality.upper():6}] {a.source}: {a.title} | {a.quality_reason}")


if __name__ == "__main__":
    main()
