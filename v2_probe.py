from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup

from v2_authors import author_prior, source_prior

UA = "Mozilla/5.0 GiantsNewsBotV2Probe/0.7"
TIMEOUT = 20
ATHLETIC_AUTHOR_ENRICH_LIMIT = 8
NBC_AUTHOR_ENRICH_LIMIT = 10

TRUSTED_RADAR_SOURCES = [
    "San Francisco Chronicle",
    "Mercury News",
    "Baseball America",
    "Associated Press",
    "KNBR",
]


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
    source_preference: str = ""
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
    "injuries & transactions",
    "trivia puzzle",
    "top 30 prospects list",
    "top 100 and top 30 prospects lists are here",
    "power rankings",
    "each team's",
    "every team",
    "for every team",
    "all 30 teams",
)

DERIVATIVE_PATTERNS = (
    "where giants' farm system ranks in",
    "where the giants' farm system ranks in",
)

# Free-viewing/event ads sometimes arrive through an otherwise legitimate team
# RSS feed and look article-like enough to pass the generic title gate.
PROMOTIONAL_TITLE_RE = re.compile(
    r"\bwatch\b.*\b(?:prospect|game)\b.*\bfor free\b|"
    r"\bplay at (?:low-|high-)?a\b.*\bfor free\b",
    flags=re.I,
)

# NBC frequently turns a broadcaster's on-air reaction into a short article.
# Keep actual broadcaster news/features, but suppress the low-information
# "X believes/reacts/thinks" repackaging class.
NBC_BROADCASTER_REACTION_RE = re.compile(
    r"^(?:mike krukow|duane kuiper|jon miller|dave flemming)\b.*\b"
    r"(?:believes?|thinks?|reacts?|weighs in|shares (?:his )?thoughts|explains why)\b",
    flags=re.I,
)

GAME_STORY_PATTERNS = (
    "what we learned",
    "observations",
    " in win",
    " in loss",
    " win over ",
    " loss to ",
    " falls to ",
    " fall to ",
    " doom ",
)

RESULT_VERBS = re.compile(
    r"\b(?:lead|leads|lift|lifts|power|powers|propel|propels|beat|beats|edge|edges|"
    r"defeat|defeats|top|tops|rout|routs)\b.*\b(?:over|past)\b",
    flags=re.I,
)


def clean(text: str) -> str:
    value = text or ""
    if "<" in value and ">" in value:
        value = BeautifulSoup(value, "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", value).strip()


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


def _plausible_author(value: str) -> str:
    text = re.sub(r"^\s*by\s+", "", clean(value), flags=re.I).strip()
    if not text or text.startswith("http://") or text.startswith("https://"):
        return ""
    words = text.split()
    if not 2 <= len(words) <= 7 or len(text) > 100:
        return ""
    if text.lower() in {
        "nbc sports bay area",
        "the athletic",
        "major league baseball",
        "san francisco giants",
    }:
        return ""
    return text


def extract_card_author(anchor) -> str:
    node = anchor.find_parent("article") or anchor.find_parent("li") or anchor.parent
    if not node:
        return ""
    for selector in ('[rel="author"]', '[class*="author"]', '[class*="byline"]'):
        for candidate in node.select(selector):
            author = _plausible_author(candidate.get_text(" "))
            if author:
                return author
    return ""


def _author_from_json_value(value) -> str:
    if isinstance(value, str):
        return _plausible_author(value)
    if isinstance(value, dict):
        return _plausible_author(value.get("name", ""))
    if isinstance(value, list):
        for item in value:
            author = _author_from_json_value(item)
            if author:
                return author
    return ""


def _iter_json_nodes(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_json_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_json_nodes(child)


def structured_meta_author(url: str) -> str:
    """Best-effort structured byline enrichment; failure never blocks discovery."""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": UA},
            timeout=TIMEOUT,
            allow_redirects=True,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        for attrs in (
            {"name": "author"},
            {"name": "parsely-author"},
            {"property": "article:author"},
            {"name": "byl"},
        ):
            tag = soup.find("meta", attrs=attrs)
            if tag and tag.get("content"):
                author = _plausible_author(tag.get("content", ""))
                if author:
                    return author

        for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
            raw = script.string or script.get_text("", strip=True)
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            for node in _iter_json_nodes(payload):
                node_type = node.get("@type", "")
                types = node_type if isinstance(node_type, list) else [node_type]
                if not any("article" in str(item).lower() for item in types):
                    continue
                author = _author_from_json_value(node.get("author"))
                if author:
                    return author

        for selector in (
            '[class*="byline"] a',
            '[class*="byline"]',
            '[class*="author"] a',
            '[data-testid*="author"]',
        ):
            for node in soup.select(selector):
                author = _plausible_author(node.get_text(" "))
                if author:
                    return author
    except Exception:
        return ""
    return ""


def classify(source: str, title: str, author: str = "") -> tuple[str, str, str]:
    blob = title.lower()
    prior = author_prior(author)
    preference = prior["preference"] if prior else ""

    if PROMOTIONAL_TITLE_RE.search(title):
        return "low", "promotional_content", preference
    if source == "MLB.com" and not re.search(r"\bmaria guardado\b", author or "", flags=re.I):
        return "low", "mlb_non_guardado", preference
    if source == "NBC Sports Bay Area" and NBC_BROADCASTER_REACTION_RE.search(title):
        return "low", "broadcaster_quote_repackaging", preference
    if any(pattern in blob for pattern in LOW_VALUE_PATTERNS):
        return "low", "commodity_or_generic_content", preference
    if any(pattern in blob for pattern in DERIVATIVE_PATTERNS):
        return "low", "summarizes_other_publication", preference
    if source == "FanGraphs" and blob.startswith("sunday notes:"):
        return "low", "broad_recurring_roundup", preference

    if any(pattern in blob for pattern in GAME_STORY_PATTERNS) or RESULT_VERBS.search(title):
        return "medium", "game_story_or_postgame_analysis", preference

    if preference == "elite":
        return "high", "elite_author", preference
    if preference:
        return "high", f"known_author:{preference}", preference
    return "high", "original_news_or_analysis_candidate", preference


def parse_feed(url: str):
    feed = feedparser.parse(url, request_headers={"User-Agent": UA})
    status = getattr(feed, "status", None)
    if status and status >= 400:
        raise RuntimeError(f"feed HTTP {status}: {url}")
    if getattr(feed, "bozo", False) and not feed.entries:
        raise RuntimeError(f"invalid feed {url}: {getattr(feed, 'bozo_exception', '')}")
    return feed


def make_article(
    *,
    source: str,
    title: str,
    url: str,
    published: str = "",
    author: str = "",
    summary: str = "",
    section: str = "",
    access: str = "unknown",
) -> Article:
    quality, reason, preference = classify(source, title, author)
    publication_prior = source_prior(source)
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
        source_preference=publication_prior["preference"] if publication_prior else "",
        quality=quality,
        quality_reason=reason,
    )


def articles_from_feed(
    *,
    source: str,
    feed_url: str,
    section: str,
    access: str,
    limit: int = 40,
    require_giants_relevance: bool = True,
) -> list[Article]:
    feed = parse_feed(feed_url)
    out: list[Article] = []
    seen: set[str] = set()
    for entry in feed.entries[:limit]:
        title = clean(getattr(entry, "title", ""))
        summary = clean(getattr(entry, "summary", ""))
        url = getattr(entry, "link", "") or ""
        if not title or not url or url in seen:
            continue
        if require_giants_relevance and not giants_relevant(f"{title} {summary}"):
            continue
        seen.add(url)
        out.append(make_article(
            source=source,
            title=title,
            url=url,
            published=getattr(entry, "published", "") or getattr(entry, "updated", "") or "",
            author=entry_author(entry),
            summary=summary,
            section=section,
            access=access,
        ))
    return out


def discover_sf_standard() -> list[Article]:
    return articles_from_feed(
        source="San Francisco Standard",
        feed_url="https://sfstandard.com/category/sports/feed/",
        section="Sports RSS",
        access="free",
        limit=30,
    )


def discover_athletic() -> list[Article]:
    feed = parse_feed("https://www.nytimes.com/athletic/rss/mlb/sf-giants/")
    out: list[Article] = []
    seen: set[str] = set()
    enriched = 0
    for entry in feed.entries[:60]:
        title = clean(getattr(entry, "title", ""))
        summary = clean(getattr(entry, "summary", ""))
        url = getattr(entry, "link", "") or ""
        if not title or not url or url in seen or not giants_relevant(f"{title} {summary}"):
            continue
        seen.add(url)
        author = entry_author(entry)
        initial = make_article(
            source="The Athletic",
            title=title,
            url=url,
            published=getattr(entry, "published", "") or getattr(entry, "updated", "") or "",
            author=author,
            summary=summary,
            section="Giants RSS",
            access="paywalled",
        )
        if not author and initial.quality != "low" and enriched < ATHLETIC_AUTHOR_ENRICH_LIMIT:
            author = structured_meta_author(url)
            enriched += 1
            if author:
                initial = make_article(
                    source="The Athletic",
                    title=title,
                    url=url,
                    published=initial.published,
                    author=author,
                    summary=summary,
                    section="Giants RSS + author meta",
                    access="paywalled",
                )
        out.append(initial)
    return out


def discover_mlb() -> list[Article]:
    return articles_from_feed(
        source="MLB.com",
        feed_url="https://www.mlb.com/giants/feeds/news/rss.xml",
        section="Official Giants RSS",
        access="free",
        limit=30,
        require_giants_relevance=False,
    )


def discover_sfgate() -> list[Article]:
    return articles_from_feed(
        source="SFGATE",
        feed_url="https://www.sfgate.com/sports/feed/San-Francisco-Giants-RSS-Feed-428.php",
        section="Giants RSS",
        access="free",
        limit=30,
        require_giants_relevance=False,
    )


def discover_fangraphs() -> list[Article]:
    return articles_from_feed(
        source="FanGraphs",
        feed_url="https://blogs.fangraphs.com/category/teams/giants/feed/",
        section="Giants category RSS",
        access="free",
        limit=20,
    )


def discover_nbc() -> list[Article]:
    pages = [
        ("Giants News", "https://www.nbcsportsbayarea.com/mlb/san-francisco-giants/giants-news/"),
        ("Giants Analysis", "https://www.nbcsportsbayarea.com/mlb/san-francisco-giants/giants-analysis/"),
    ]
    seen: set[str] = set()
    out: list[Article] = []
    author_enrichments = 0
    for section, page_url in pages:
        response = requests.get(page_url, headers={"User-Agent": UA}, timeout=TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        for anchor in soup.find_all("a", href=True):
            href = urljoin(page_url, anchor["href"])
            path = urlparse(href).path.rstrip("/")
            title = clean(anchor.get_text(" "))
            parts = [part for part in path.split("/") if part]
            if len(parts) < 4 or parts[:2] != ["mlb", "san-francisco-giants"]:
                continue
            if not parts[-1].isdigit() or "video" in parts:
                continue
            if not title or len(title) < 20 or href in seen:
                continue
            seen.add(href)
            author = extract_card_author(anchor)
            if not author and author_enrichments < NBC_AUTHOR_ENRICH_LIMIT:
                author = structured_meta_author(href)
                author_enrichments += 1
            out.append(make_article(
                source="NBC Sports Bay Area",
                title=title,
                url=href,
                author=author,
                section=section + (" + author meta" if author else ""),
                access="free",
            ))
    return out


def main() -> None:
    discoverers = [
        discover_sf_standard,
        discover_athletic,
        discover_mlb,
        discover_sfgate,
        discover_fangraphs,
        discover_nbc,
    ]

    all_articles: list[Article] = []
    health: dict[str, dict] = {}
    for discover in discoverers:
        name = discover.__name__.replace("discover_", "")
        try:
            items = discover()
            health[name] = {"ok": True, "count": len(items)}
            all_articles.extend(items)
        except Exception as exc:
            health[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    articles = list({article.url: article for article in all_articles}.values())
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "health": health,
        "trusted_radar_sources": TRUSTED_RADAR_SOURCES,
        "counts_by_quality": {
            quality: sum(1 for article in articles if article.quality == quality)
            for quality in ("high", "medium", "low")
        },
        "articles": [asdict(article) for article in articles],
    }
    with open("v2-probe.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(json.dumps(health, indent=2))
    print(f"TRUSTED_RADAR: {', '.join(TRUSTED_RADAR_SOURCES)}")
    for article in articles:
        byline = f" · {article.author}" if article.author else ""
        access = " ($)" if article.access == "paywalled" else ""
        source_weight = f" source={article.source_preference}" if article.source_preference else ""
        print(
            f"[{article.quality.upper():6}] {article.source}{access}{byline}: "
            f"{article.title} | {article.quality_reason}{source_weight}"
        )


if __name__ == "__main__":
    main()
