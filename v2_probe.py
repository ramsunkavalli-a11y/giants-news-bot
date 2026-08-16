from __future__ import annotations

import gzip
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup

from v2_authors import author_prior

UA = "Mozilla/5.0 GiantsNewsBotV2Probe/0.2"
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
    access: str = "unknown"
    author_preference: str = ""
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
    "injuries and roster moves",
)

GENERIC_PATTERNS = (
    "power rankings",
    "each team's",
    "every team",
    "for every team",
    "all 30 teams",
)

GAME_RECAP_PATTERNS = (
    "what we learned",
    "observations",
    " in win",
    " in loss",
    " win over ",
    " loss to ",
    " beat ",
    " beats ",
    " rout ",
    " falls to ",
    " fall to ",
    " doom ",
)

SFGATE_REWRITE_PATTERNS = (
    "reportedly",
    "top mlb insider",
    "insider says",
    "insider believes",
    "report says",
)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def giants_relevant(text: str) -> bool:
    t = clean(text).lower()
    return "giants" in t or "san francisco" in t


def entry_author(entry) -> str:
    author = clean(getattr(entry, "author", ""))
    if author:
        return author
    authors = getattr(entry, "authors", None) or []
    if authors and isinstance(authors[0], dict):
        return clean(authors[0].get("name", ""))
    return ""


def extract_card_author(anchor) -> str:
    node = anchor.find_parent("article") or anchor.find_parent("li") or anchor.parent
    if not node:
        return ""
    for selector in ('[rel="author"]', '[class*="author"]', '[class*="byline"]'):
        for candidate in node.select(selector):
            text = re.sub(r"^\s*by\s+", "", clean(candidate.get_text(" ")), flags=re.I)
            if 1 < len(text.split()) <= 5 and len(text) < 80:
                return text
    text = clean(node.get_text(" "))[:600]
    match = re.search(
        r"\bBy\s+([A-Z][A-Za-z.'’\-]+(?:\s+[A-Z][A-Za-z.'’\-]+){1,3})(?=\s|$)",
        text,
    )
    return clean(match.group(1)) if match else ""


def classify(source: str, title: str, author: str = "", section: str = "") -> tuple[str, str, str]:
    blob = f"{title} {section}".lower()
    prior = author_prior(author)
    preference = prior["preference"] if prior else ""

    if any(p in blob for p in LOW_VALUE_PATTERNS):
        return "low", "commodity_or_recurring_content", preference
    if any(p in blob for p in GENERIC_PATTERNS):
        return "low", "generic_multi_team_content", preference
    if source == "MLB.com" and title.lower().startswith("release:"):
        return "low", "press_release", preference
    if source == "SFGATE" and any(p in blob for p in SFGATE_REWRITE_PATTERNS):
        return "low", "likely_rewrite_or_aggregation", preference
    if source == "NBC Sports Bay Area" and "baseball america" in blob and "rank" in blob:
        return "low", "summarizes_other_publication", preference

    if preference == "elite":
        return "high", "elite_author", preference

    if any(p in blob for p in GAME_RECAP_PATTERNS):
        return "medium", "game_story_or_postgame_analysis", preference

    if preference:
        return "high", f"known_author:{preference}", preference
    return "high", "original_news_or_analysis_candidate", preference


def get(url: str) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
    r.raise_for_status()
    return r


def parse_feed(url: str):
    feed = feedparser.parse(url, request_headers={"User-Agent": UA})
    status = getattr(feed, "status", None)
    if status and status >= 400:
        raise RuntimeError(f"feed HTTP {status}: {url}")
    if getattr(feed, "bozo", False) and not feed.entries:
        raise RuntimeError(f"invalid feed {url}: {getattr(feed, 'bozo_exception', '')}")
    return feed


def make_article(
    *, source: str, title: str, url: str, published: str = "", author: str = "",
    summary: str = "", section: str = "", access: str = "unknown"
) -> Article:
    quality, reason, preference = classify(source, title, author, section)
    return Article(
        source=source,
        title=title,
        url=url,
        published=published,
        author=author,
        summary=summary,
        section=section,
        access=access,
        author_preference=preference,
        quality=quality,
        quality_reason=reason,
    )


def discover_sf_standard() -> list[Article]:
    feed = parse_feed("https://sfstandard.com/category/sports/feed/")
    out: list[Article] = []
    for e in feed.entries[:30]:
        title = clean(getattr(e, "title", ""))
        summary = clean(getattr(e, "summary", ""))
        if not giants_relevant(f"{title} {summary}"):
            continue
        out.append(make_article(
            source="San Francisco Standard",
            title=title,
            url=getattr(e, "link", "") or "",
            published=getattr(e, "published", "") or getattr(e, "updated", "") or "",
            author=entry_author(e),
            summary=summary,
            section="Sports RSS",
            access="free",
        ))
    return out


def discover_athletic() -> list[Article]:
    feed = parse_feed("https://www.nytimes.com/athletic/rss/mlb/sf-giants/")
    out: list[Article] = []
    for e in feed.entries[:40]:
        title = clean(getattr(e, "title", ""))
        if not title:
            continue
        out.append(make_article(
            source="The Athletic",
            title=title,
            url=getattr(e, "link", "") or "",
            published=getattr(e, "published", "") or getattr(e, "updated", "") or "",
            author=entry_author(e),
            summary=clean(getattr(e, "summary", "")),
            section="Giants RSS",
            access="paywalled",
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
        out.append(make_article(
            source="MLB.com",
            title=title,
            url=loc,
            published=published,
            section="48-hour news sitemap",
            access="free",
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
            out.append(make_article(
                source="NBC Sports Bay Area",
                title=title,
                url=href,
                author=extract_card_author(a),
                section=section,
                access="free",
            ))
    return out


def discover_chronicle() -> list[Article]:
    page_url = "https://www.sfchronicle.com/sports/giants/"
    soup = BeautifulSoup(get(page_url).text, "lxml")
    seen: set[str] = set()
    out: list[Article] = []
    for a in soup.find_all("a", href=True):
        href = urljoin(page_url, a["href"])
        title = clean(a.get_text(" "))
        if "/sports/giants/article/" not in href:
            continue
        if not title or len(title) < 20 or href in seen:
            continue
        seen.add(href)
        out.append(make_article(
            source="San Francisco Chronicle",
            title=title,
            url=href,
            author=extract_card_author(a),
            section="Giants landing page",
            access="unknown",
        ))
    return out


def discover_sfgate() -> list[Article]:
    page_url = "https://www.sfgate.com/giants/"
    soup = BeautifulSoup(get(page_url).text, "lxml")
    seen: set[str] = set()
    out: list[Article] = []
    for a in soup.find_all("a", href=True):
        href = urljoin(page_url, a["href"])
        title = clean(a.get_text(" "))
        if "/giants/article/" not in href:
            continue
        if not title or len(title) < 20 or href in seen:
            continue
        seen.add(href)
        out.append(make_article(
            source="SFGATE",
            title=title,
            url=href,
            author=extract_card_author(a),
            section="Giants landing page",
            access="free",
        ))
    return out


def discover_fangraphs() -> list[Article]:
    page_url = "https://blogs.fangraphs.com/category/teams/giants/"
    soup = BeautifulSoup(get(page_url).text, "lxml")
    seen: set[str] = set()
    out: list[Article] = []
    for heading in soup.find_all(["h2", "h3"]):
        a = heading.find("a", href=True)
        if not a:
            continue
        href = urljoin(page_url, a["href"])
        title = clean(a.get_text(" "))
        if urlparse(href).netloc not in {"blogs.fangraphs.com", "www.fangraphs.com"}:
            continue
        if not title or len(title) < 20 or href in seen or not giants_relevant(title):
            continue
        seen.add(href)
        out.append(make_article(
            source="FanGraphs",
            title=title,
            url=href,
            author=extract_card_author(a),
            section="Giants archive",
            access="unknown",
        ))
    return out


def discover_baseball_america() -> list[Article]:
    out: list[Article] = []
    seen: set[str] = set()

    try:
        feed = parse_feed("https://www.baseballamerica.com/feed/")
        for e in feed.entries[:100]:
            title = clean(getattr(e, "title", ""))
            if not title or not giants_relevant(title):
                continue
            href = getattr(e, "link", "") or ""
            if not href or href in seen:
                continue
            seen.add(href)
            out.append(make_article(
                source="Baseball America",
                title=title,
                url=href,
                published=getattr(e, "published", "") or getattr(e, "updated", "") or "",
                author=entry_author(e),
                summary=clean(getattr(e, "summary", "")),
                section="site RSS",
                access="unknown",
            ))
    except Exception:
        pass

    if out:
        return out

    page_url = "https://www.baseballamerica.com/stories/teams/2019-san-francisco-giants/"
    try:
        soup = BeautifulSoup(get(page_url).text, "lxml")
    except requests.RequestException:
        return out
    for heading in soup.find_all(["h2", "h3", "h4"]):
        a = heading.find("a", href=True)
        if not a:
            continue
        href = urljoin(page_url, a["href"])
        title = clean(a.get_text(" "))
        if "/stories/" not in href or not title or len(title) < 20 or href in seen:
            continue
        if not giants_relevant(title):
            continue
        seen.add(href)
        out.append(make_article(
            source="Baseball America",
            title=title,
            url=href,
            author=extract_card_author(a),
            section="Giants team page",
            access="unknown",
        ))
    return out


def main() -> None:
    discoverers = [
        discover_sf_standard,
        discover_athletic,
        discover_mlb,
        discover_nbc,
        discover_chronicle,
        discover_sfgate,
        discover_fangraphs,
        discover_baseball_america,
    ]
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
        byline = f" · {a.author}" if a.author else ""
        access = " ($)" if a.access == "paywalled" else ""
        print(
            f"[{a.quality.upper():6}] {a.source}{access}{byline}: {a.title} "
            f"| {a.quality_reason}"
        )


if __name__ == "__main__":
    main()
