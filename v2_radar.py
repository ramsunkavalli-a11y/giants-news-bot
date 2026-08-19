from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from urllib.parse import urlencode, urlparse

import feedparser

try:
    from googlenewsdecoder import gnewsdecoder
except Exception:
    gnewsdecoder = None

from v2_authors import normalize_author
from v2_probe import Article, clean, make_article, structured_meta_author

UA = "Mozilla/5.0 GiantsNewsBotV2Radar/1.3"
RADAR_MAX_PER_AUTHOR = 6


@dataclass(frozen=True)
class RadarTarget:
    author: str
    source: str
    domain: str


CORE_WRITER_RADAR_TARGETS = (
    RadarTarget("Susan Slusser", "San Francisco Chronicle", "sfchronicle.com"),
    RadarTarget("Shayna Rubin", "San Francisco Chronicle", "sfchronicle.com"),
    RadarTarget("Justice delos Santos", "Mercury News", "mercurynews.com"),
)

# Google News occasionally returns publisher section/listing pages for an
# otherwise precise author query. Those pages can inherit snippets that contain
# the author and "Giants" even though the result itself is not an article.
GENERIC_PAGINATION_TITLE_RE = re.compile(r"(?:^|[-–—]\s+)page\s+\d+\s*$", flags=re.I)


def google_news_rss_url(target: RadarTarget, hours_back: int = 72) -> str:
    query = f'site:{target.domain} "{target.author}" "Giants" when:{hours_back}h'
    params = urlencode({"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    return f"https://news.google.com/rss/search?{params}"


def domain_matches(url: str, domain: str) -> bool:
    host = urlparse(url or "").netloc.lower().split(":", 1)[0]
    domain = domain.lower()
    return host == domain or host.endswith("." + domain)


def radar_url_allowed(target: RadarTarget, url: str) -> bool:
    """Require a publisher-specific article surface, not merely the right domain."""
    if not domain_matches(url, target.domain):
        return False

    path = urlparse(url or "").path or "/"
    if target.domain.lower() == "sfchronicle.com":
        # The Chronicle radar exists specifically for Giants beat coverage.
        # Known valid results use this article subtree; weather, homepage,
        # pagination, and other section pages must never inherit author credit.
        return path.startswith("/sports/giants/")

    return True


def radar_title_allowed(title: str) -> bool:
    value = clean(title)
    if not value:
        return False
    if GENERIC_PAGINATION_TITLE_RE.search(value):
        return False
    return True


def decode_google_news_url(url: str) -> str:
    """Contained decoder: failure returns empty and never affects direct adapters."""
    if not gnewsdecoder or not url:
        return ""
    try:
        value = gnewsdecoder(url)
        if not isinstance(value, dict) or not value.get("status"):
            return ""
        decoded = str(value.get("decoded_url") or "").strip()
        if not decoded.startswith("https://"):
            return ""
        return decoded
    except Exception:
        return ""


def strip_google_source_suffix(title: str, source: str) -> str:
    value = clean(title)
    for separator in (" - ", " — ", " – "):
        suffix = f"{separator}{source}"
        if value.lower().endswith(suffix.lower()):
            return value[: -len(suffix)].strip()
    return value


def metadata_byline_includes_target(metadata_author: str, target_author: str) -> bool:
    """Accept an exact target writer inside an otherwise valid co-byline."""
    target = normalize_author(target_author)
    if not target:
        return False
    value = " ".join((metadata_author or "").strip().lower().split())
    if normalize_author(value) == target:
        return True
    value = value.replace(" & ", ",").replace(" and ", ",")
    parts = [normalize_author(part.strip()) for part in value.split(",") if part.strip()]
    return target in parts or target in value


def _feed_records(target: RadarTarget, hours_back: int) -> list[dict]:
    feed = feedparser.parse(
        google_news_rss_url(target, hours_back),
        request_headers={"User-Agent": UA},
    )
    status = getattr(feed, "status", None)
    if status and status >= 400:
        return []
    if getattr(feed, "bozo", False) and not feed.entries:
        return []

    records: list[dict] = []
    for entry in feed.entries[:RADAR_MAX_PER_AUTHOR]:
        google_url = getattr(entry, "link", "") or ""
        direct_url = decode_google_news_url(google_url)
        if not direct_url or not radar_url_allowed(target, direct_url):
            continue
        source = getattr(entry, "source", {})
        google_source = source.get("title", "") if isinstance(source, dict) else ""
        title = strip_google_source_suffix(
            getattr(entry, "title", "") or "",
            google_source or target.source,
        )
        summary = clean(getattr(entry, "summary", "") or "")
        if not radar_title_allowed(title):
            continue
        records.append({
            "target": target,
            "url": direct_url,
            "title": title,
            "summary": summary,
            "published": getattr(entry, "published", "") or getattr(entry, "updated", "") or "",
        })
    return records


def unique_author_records(records: list[dict]) -> list[dict]:
    """Keep only URLs discovered through one target query; this never proves a byline."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record.get("url", "")].append(record)

    accepted: list[dict] = []
    for url, items in grouped.items():
        authors = {
            normalize_author(item["target"].author)
            for item in items
            if item.get("target")
        }
        if not url or len(authors) != 1:
            continue
        accepted.append(items[0])
    return accepted


def discover_core_writer_radar(hours_back: int = 72) -> list[Article]:
    """Optional author-targeted discovery for blocked Chronicle/Mercury pages.

    The Google query identifies where to look, not who wrote the result. A target
    author is attached only when the article itself exposes metadata confirming
    that writer. If metadata is unavailable, the valid story can still enter the
    candidate pool with an unknown author. Contradictory metadata vetoes it.
    """
    records: list[dict] = []
    for target in CORE_WRITER_RADAR_TARGETS:
        records.extend(_feed_records(target, hours_back))

    articles: list[Article] = []
    for record in unique_author_records(records):
        target: RadarTarget = record["target"]
        metadata_author = structured_meta_author(record["url"])

        if metadata_author and not metadata_byline_includes_target(metadata_author, target.author):
            continue

        verified_author = target.author if metadata_author else ""
        articles.append(make_article(
            source=target.source,
            title=record["title"],
            url=record["url"],
            published=record["published"],
            author=verified_author,
            summary=record["summary"],
            section=(
                "Google News core-writer radar + verified author meta"
                if verified_author
                else "Google News core-writer radar (author unverified)"
            ),
            access="unknown",
        ))

    # Decoder/query duplication should not leak into the rest of V2.
    return list({article.url: article for article in articles}.values())