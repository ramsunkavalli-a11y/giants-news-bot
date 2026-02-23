#!/usr/bin/env python3
"""
SF Giants news -> Bluesky bot.

Configuration (env vars):
- BSKY_IDENTIFIER (required unless DRY_RUN=1)
- BSKY_APP_PASSWORD (required unless DRY_RUN=1)
- BSKY_PDS (default: https://bsky.social)
- STATE_FILE (default: state.json)
- MAX_POSTS_PER_RUN (default: 10)
- MAX_PER_SOURCE_PER_RUN (default: 3)
- HOURS_BACK (default: 24)
- KEEP_POSTED_DAYS (default: 21)
- META_CACHE_DAYS (default: KEEP_POSTED_DAYS)
- DRY_RUN (default: 0)
- USER_AGENT (optional)
- REQUEST_TIMEOUT (default: 15)
- MAX_NON_RSS_URLS_PER_SOURCE (default: 60)
- MAX_RSS_ENTRIES_PER_FEED (default: 40)
- MAX_VALIDATION_TARGET (default: max(20, MAX_POSTS_PER_RUN*3))
- AUTHOR_PRIORITY_JSON (optional): {"high": ["..."]}
"""

from __future__ import annotations

import html as html_lib
import json
import os
import re
import time
import html as html_lib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import feedparser
import requests
from dateutil import parser as dtparser
from requests.adapters import HTTPAdapter

UA = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 GiantsNewsBot/2026",
)
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))
DRY_RUN = os.getenv("DRY_RUN", "0").lower() in {"1", "true", "yes"}
STATE_FILE = os.getenv("STATE_FILE", "state.json")
MAX_POSTS_PER_RUN = int(os.getenv("MAX_POSTS_PER_RUN", "10"))
MAX_PER_SOURCE_PER_RUN = int(os.getenv("MAX_PER_SOURCE_PER_RUN", "3"))
HOURS_BACK = int(os.getenv("HOURS_BACK", "24"))
KEEP_POSTED_DAYS = int(os.getenv("KEEP_POSTED_DAYS", "21"))
META_CACHE_DAYS = int(os.getenv("META_CACHE_DAYS", str(KEEP_POSTED_DAYS)))
MAX_NON_RSS_URLS_PER_SOURCE = int(os.getenv("MAX_NON_RSS_URLS_PER_SOURCE", "60"))
MAX_RSS_ENTRIES_PER_FEED = int(os.getenv("MAX_RSS_ENTRIES_PER_FEED", "40"))
MAX_VALIDATION_TARGET = int(os.getenv("MAX_VALIDATION_TARGET", str(max(20, MAX_POSTS_PER_RUN * 3))))

BSKY_PDS = os.getenv("BSKY_PDS", "https://bsky.social")
BSKY_IDENTIFIER = os.getenv("BSKY_IDENTIFIER", "")
BSKY_APP_PASSWORD = os.getenv("BSKY_APP_PASSWORD", "")

RSS_SOURCES = [
    ("SF Standard", "https://sfstandard.com/category/sports/feed/"),
    ("SFGate Giants", "https://www.sfgate.com/giants/feed/Giants-447.php"),
    ("NYTimes Baseball", "https://rss.nytimes.com/services/xml/rss/nyt/Baseball.xml"),
]

GOOGLE_NEWS_QUERIES = {
    "NBC Sports Bay Area": 'site:nbcsportsbayarea.com "San Francisco Giants"',
    "SF Chronicle Giants": 'site:sfchronicle.com "San Francisco Giants"',
    "Mercury News Giants": 'site:mercurynews.com "San Francisco Giants"',
    "AP Giants": 'site:apnews.com "San Francisco Giants"',
    "MLB Giants": 'site:mlb.com/giants/news "San Francisco Giants"',
    "Fangraphs Giants": 'site:blogs.fangraphs.com giants',
    "Baseball America Giants": 'site:baseballamerica.com "San Francisco Giants"',
    "KNBR Giants": 'site:knbr.com giants',
}

NON_RSS_LISTING_URLS = {
    "NBC Sports Bay Area": "https://www.nbcsportsbayarea.com/mlb/san-francisco-giants/",
    "SF Chronicle Giants": "https://www.sfchronicle.com/sports/giants/",
    "Mercury News Giants": "https://www.mercurynews.com/tag/san-francisco-giants/",
    "AP Giants hub": "https://apnews.com/hub/san-francisco-giants",
    "MLB Giants News": "https://www.mlb.com/giants/news",
    "SFGiants News": "https://www.mlb.com/giants/news",
    "Fangraphs Giants": "https://blogs.fangraphs.com/category/giants/",
    "Baseball America Giants": "https://www.baseballamerica.com/teams/2003/san-francisco-giants/",
    "KNBR Giants": "https://www.knbr.com/category/giants/",
}

AGGREGATOR_BLOCKLIST = {
    "news.google.com",
    "google.com",
    "feedly.com",
    "feedspot.com",
    "newsbreak.com",
}

TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "ref",
    "refsrc",
    "mc_cid",
    "mc_eid",
    "igshid",
    "source",
}

STORY_REJECT_SEGMENTS = {
    "tag",
    "tags",
    "topic",
    "topics",
    "category",
    "categories",
    "section",
    "sections",
    "hub",
}

STORY_REJECT_SLUGS = {
    "giants-news",
    "giants-rumors",
    "giants-playoffs",
    "giants-spring-training",
    "san-francisco-giants",
    "sf-giants",
    "giants",
    "news",
    "sports",
    "schedule",
    "tickets",
    "roster",
    "stats",
    "standings",
    "depth-chart",
    "video",
}

BASEBALL_TERMS = {
    "mlb", "baseball", "pitcher", "inning", "oracle park", "nl west", "rotation",
    "bullpen", "hitter", "san francisco giants", "sf giants", "giants",
}
NFL_TERMS = {"new york giants", "quarterback", "touchdown", "nfl", "super bowl"}

DEFAULT_HIGH_AUTHORS = [
    "Justice delos Santos",
    "Alex Pavlovic",
    "John Shea",
    "Shayna Rubin",
    "Susan Slusser",
    "Janie McCauley",
    "Andrew Baggarly",
    "Grant Brisbee",
]

META_TITLE_RE = re.compile(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', re.I)
META_AUTHOR_RE = re.compile(r'<meta[^>]+name=["\']author["\'][^>]+content=["\']([^"\']+)', re.I)
META_ARTICLE_AUTHOR_RE = re.compile(r'<meta[^>]+property=["\']article:author["\'][^>]+content=["\']([^"\']+)', re.I)
META_PARSELY_AUTHOR_RE = re.compile(r'<meta[^>]+name=["\']parsely-author["\'][^>]+content=["\']([^"\']+)', re.I)
META_BYL_RE = re.compile(r'<meta[^>]+name=["\']byl["\'][^>]+content=["\']([^"\']+)', re.I)
CANONICAL_RE = re.compile(r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)', re.I)
OG_URL_RE = re.compile(r'<meta[^>]+property=["\']og:url["\'][^>]+content=["\']([^"\']+)', re.I)
TITLE_TAG_RE = re.compile(r"<title>(.*?)</title>", re.I | re.S)
A_HREF_RE = re.compile(r"<a[^>]+href=[\"']([^\"']+)[\"']", re.I)
NEXT_DATA_RE = re.compile(r'<script[^>]*id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>', re.I | re.S)
JSON_LD_RE = re.compile(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.I | re.S)
META_DESC_RE = re.compile(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)', re.I)
META_TWITTER_DESC_RE = re.compile(r'<meta[^>]+name=["\']twitter:description["\'][^>]+content=["\']([^"\']+)', re.I)
META_IMAGE_RE = re.compile(r'<meta[^>]+property=["\']og:image(?::url)?["\'][^>]+content=["\']([^"\']+)', re.I)
META_TWITTER_IMAGE_RE = re.compile(r'<meta[^>]+name=["\']twitter:image(?::src)?["\'][^>]+content=["\']([^"\']+)', re.I)
STRIP_TAGS_RE = re.compile(r"<[^>]+>")


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})
SESSION.mount("http://", HTTPAdapter(pool_connections=20, pool_maxsize=20))
SESSION.mount("https://", HTTPAdapter(pool_connections=20, pool_maxsize=20))


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})
SESSION.mount("http://", HTTPAdapter(pool_connections=20, pool_maxsize=20))
SESSION.mount("https://", HTTPAdapter(pool_connections=20, pool_maxsize=20))


@dataclass
class Candidate:
    source: str
    url: str
    title: str = ""
    author: str = ""
    summary: str = ""
    categories: List[str] = None
    image_url: str = ""

    def __post_init__(self) -> None:
        if self.categories is None:
            self.categories = []


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def log(msg: str) -> None:
    print(f"[{datetime.utcnow().isoformat()}] {msg}")


def clean_text(text: str) -> str:
    if not text:
        return ""
    txt = STRIP_TAGS_RE.sub(" ", text)
    txt = html_lib.unescape(txt)
    return re.sub(r"\s+", " ", txt).strip()


def normalize_author(s: str) -> str:
    s = clean_text(s or "").lower().strip()
    if not s:
        return ""
    if s.startswith("by "):
        s = s[3:].strip()
    for sep in ["|", "•", "·", " - ", " — ", ";"]:
        if sep in s:
            s = s.split(sep, 1)[0].strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            s = f"{parts[1]} {parts[0]}"
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_high_authors() -> Set[str]:
    high = DEFAULT_HIGH_AUTHORS[:]
    raw = os.getenv("AUTHOR_PRIORITY_JSON", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            maybe = parsed.get("high", []) if isinstance(parsed, dict) else []
            if isinstance(maybe, list):
                high = [str(x) for x in maybe if str(x).strip()]
        except Exception:
            pass
    return {normalize_author(a) for a in high if normalize_author(a)}


HIGH_AUTHORS = load_high_authors()


def is_high_author(author: str) -> bool:
    return normalize_author(author) in HIGH_AUTHORS


def canonicalize_url(url: Any) -> str:
    if not isinstance(url, str):
        return ""
    if not url:
        return ""
    url = url.strip()
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    filtered = []
    for k, v in parse_qsl(parsed.query, keep_blank_values=True):
        kl = k.lower()
        if kl in TRACKING_QUERY_KEYS or kl.startswith("utm_") or kl.startswith("mc_"):
            continue
        filtered.append((k, v))

    cleaned = parsed._replace(netloc=host, fragment="", query=urlencode(filtered, doseq=True))
    return urlunparse(cleaned)


def is_bad_domain(domain: str) -> bool:
    d = (domain or "").lower()
    return d in AGGREGATOR_BLOCKLIST or d == "google.com" or d.endswith(".google.com")


def parse_dt_or_none(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = dtparser.isoparse(s)
    except Exception:
        try:
            dt = dtparser.parse(s)
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def resolve_final_url(url: str, redirect_cache: Dict[str, str]) -> str:
    c = canonicalize_url(url)
    if not c:
        return ""
    if c in redirect_cache:
        cached = redirect_cache[c]
        if isinstance(cached, dict):
            final = canonicalize_url(cached.get("final", ""))
            if final:
                redirect_cache[c] = final
                return final
        elif isinstance(cached, str):
            final = canonicalize_url(cached)
            if final:
                redirect_cache[c] = final
                return final
    final = c
    try:
        r = SESSION.request("HEAD", c, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        final = canonicalize_url(r.url or c)
        if r.status_code >= 400:
            with SESSION.get(c, timeout=REQUEST_TIMEOUT, allow_redirects=True, stream=True) as fallback:
                final = canonicalize_url(fallback.url or c)
    except Exception:
        pass
    redirect_cache[c] = final
    return final


def is_story_url(url: str) -> bool:
    try:
        p = urlparse(url)
    except Exception:
        return False
    if p.scheme not in {"http", "https"} or not p.netloc:
        return False
    if is_bad_domain(p.netloc):
        return False

    path = (p.path or "/").strip("/").lower()
    if not path:
        return False

    segs = [s for s in path.split("/") if s]
    if len(segs) <= 1:
        return False

    if any(seg in STORY_REJECT_SEGMENTS for seg in segs):
        return False

    last = segs[-1]
    if last in STORY_REJECT_SLUGS:
        return False

    if re.search(r"/20\d{2}/\d{1,2}/\d{1,2}/", "/" + path + "/"):
        return True
    if re.search(r"20\d{2}-\d{2}-\d{2}", path):
        return True
    if last.endswith(".html"):
        return True
    if "/article/" in "/" + path + "/":
        return True

    if p.netloc.endswith("mlb.com"):
        if path.startswith("giants/news/") or path.startswith("news/"):
            return "-" in last and len(last) >= 20 and last not in STORY_REJECT_SLUGS

    if p.netloc.endswith("nbcsportsbayarea.com"):
        if last in {"giants-news", "giants-rumors", "giants-playoffs", "giants-spring-training"}:
            return False
        if len(segs) >= 3 and segs[-1].isdigit() and len(segs[-1]) >= 6 and "-" in segs[-2]:
            return True

    return "-" in last and len(last) >= 24 and last not in STORY_REJECT_SLUGS


def giants_relevant(title: str, summary: str, categories: Iterable[str], url: str) -> bool:
    text = " ".join([clean_text(title or ""), clean_text(summary or ""), " ".join(categories or []), url or ""]).lower()
    if any(term in text for term in NFL_TERMS):
        return False
    if "san francisco giants" in text or "sf giants" in text:
        return True
    if "giants" in text and any(term in text for term in BASEBALL_TERMS):
        return True
    return False


def fetch_html(url: str) -> Optional[str]:
    try:
        r = SESSION.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400 or "text/html" not in r.headers.get("Content-Type", ""):
            return None
        return r.text[:2_000_000]
    except Exception:
        return None


def clean_text(text: str) -> str:
    if not text:
        return ""
    txt = STRIP_TAGS_RE.sub(" ", text)
    txt = html_lib.unescape(txt)
    return re.sub(r"\s+", " ", txt).strip()


def extract_meta(url: str, html: str) -> Tuple[str, str, str, str, str]:
    title = ""
    author = ""
    canonical = ""
    description = ""
    image_url = ""

    m = META_TITLE_RE.search(html)
    if m:
        title = clean_text(m.group(1))
    if not title:
        t = TITLE_TAG_RE.search(html)
        if t:
            title = clean_text(t.group(1))

    for pat in (META_AUTHOR_RE, META_ARTICLE_AUTHOR_RE, META_PARSELY_AUTHOR_RE, META_BYL_RE):
        ma = pat.search(html)
        if ma and not author:
            author = clean_text(ma.group(1))

    c1 = CANONICAL_RE.search(html)
    c2 = OG_URL_RE.search(html)
    canonical = (c1.group(1) if c1 else "") or (c2.group(1) if c2 else "")
    canonical = canonicalize_url(urljoin(url, canonical)) if canonical else ""

    d1 = META_DESC_RE.search(html)
    d2 = META_TWITTER_DESC_RE.search(html)
    description = clean_text((d1.group(1) if d1 else "") or (d2.group(1) if d2 else ""))

    i1 = META_IMAGE_RE.search(html)
    i2 = META_TWITTER_IMAGE_RE.search(html)
    image_url = (i1.group(1) if i1 else "") or (i2.group(1) if i2 else "")
    image_url = canonicalize_url(urljoin(url, image_url)) if image_url else ""

    for block in JSON_LD_RE.findall(html):
        try:
            data = json.loads(block)
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            if not title and item.get("headline"):
                title = str(item.get("headline")).strip()
            if not description and item.get("description"):
                description = clean_text(str(item.get("description")))
            if not image_url and item.get("image"):
                img = item.get("image")
                if isinstance(img, str):
                    image_url = canonicalize_url(urljoin(url, img))
                elif isinstance(img, list) and img:
                    first = img[0]
                    if isinstance(first, str):
                        image_url = canonicalize_url(urljoin(url, first))
                    elif isinstance(first, dict) and first.get("url"):
                        image_url = canonicalize_url(urljoin(url, str(first.get("url"))))
                elif isinstance(img, dict) and img.get("url"):
                    image_url = canonicalize_url(urljoin(url, str(img.get("url"))))
            if not author and item.get("author"):
                a = item.get("author")
                if isinstance(a, dict):
                    author = clean_text(str(a.get("name", "")))
                elif isinstance(a, list) and a and isinstance(a[0], dict):
                    author = clean_text(str(a[0].get("name", "")))

    if not author:
        sample = clean_text(html[:200_000])
        m2 = BYLINE_RE.search(sample)
        if m2:
            author = clean_text(m2.group(1))

    return title, author, canonical, description, image_url


def state_load() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {"posted_urls": {}, "redirect_cache": {}, "meta_cache": {}}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("posted_urls", {})
    data.setdefault("redirect_cache", {})
    data.setdefault("meta_cache", {})

    normalized_redirect_cache: Dict[str, str] = {}
    for raw_key, raw_val in data["redirect_cache"].items():
        key = canonicalize_url(raw_key)
        if not key:
            continue
        val = canonicalize_url(raw_val.get("final", "")) if isinstance(raw_val, dict) else canonicalize_url(raw_val)
        normalized_redirect_cache[key] = val or key
    data["redirect_cache"] = normalized_redirect_cache
    return data


def state_save(state: Dict[str, Any]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def prune_state(state: Dict[str, Any]) -> None:
    posted_cutoff = now_utc() - timedelta(days=KEEP_POSTED_DAYS)
    for url, ts in list(state["posted_urls"].items()):
        dt = parse_dt_or_none(ts)
        if not dt or dt < posted_cutoff:
            del state["posted_urls"][url]

    meta_cutoff = now_utc() - timedelta(days=META_CACHE_DAYS)
    for url, data in list(state["meta_cache"].items()):
        ts = data.get("ts", "") if isinstance(data, dict) else ""
        dt = parse_dt_or_none(ts)
        if not dt or dt < meta_cutoff:
            del state["meta_cache"][url]


def discover_from_rss() -> List[Candidate]:
    out: List[Candidate] = []
    for source_name, feed_url in RSS_SOURCES:
        feed = feedparser.parse(feed_url, request_headers={"User-Agent": UA})
        for e in feed.entries[:MAX_RSS_ENTRIES_PER_FEED]:
            published = getattr(e, "published", "") or getattr(e, "updated", "") or getattr(e, "pubDate", "") or ""
            out.append(
                Candidate(
                    source=source_name,
                    url=getattr(e, "link", "") or "",
                    title=getattr(e, "title", "") or "",
                    author=getattr(e, "author", "") or "",
                    summary=getattr(e, "summary", "") or "",
                    categories=[t.get("term", "") for t in getattr(e, "tags", []) if isinstance(t, dict)],
                    discovered_via="rss",
                    published_ts=published,
                )
            )
    return out


def google_news_rss_url(query: str) -> str:
    q = urlencode({"q": f"{query} when:{HOURS_BACK}h", "hl": "en-US", "gl": "US", "ceid": "US:en"})
    return f"https://news.google.com/rss/search?{q}"


def discover_from_google_news() -> List[Candidate]:
    out: List[Candidate] = []
    for source_name, query in GOOGLE_NEWS_QUERIES.items():
        feed = feedparser.parse(google_news_rss_url(query), request_headers={"User-Agent": UA})
        for e in feed.entries[:MAX_RSS_ENTRIES_PER_FEED]:
            src = getattr(e, "source", {})
            src_title = src.get("title", "") if isinstance(src, dict) else ""
            out.append(
                Candidate(
                    source=source_name,
                    url=getattr(e, "link", "") or "",
                    title=getattr(e, "title", "") or "",
                    author=src_title,
                    summary=getattr(e, "summary", "") or "",
                    discovered_via="google",
                    published_ts=getattr(e, "published", "") or getattr(e, "updated", "") or "",
                )
            )
    return out


def is_recent_enough(published_at: str) -> bool:
    dt = parse_dt_or_none(published_at)
    if not dt:
        return True
    return dt >= now_utc() - timedelta(hours=HOURS_BACK)


def fetch_xml(url: str) -> Optional[str]:
    try:
        r = SESSION.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400:
            return None
        return r.text
    except Exception:
        return None


def discover_sitemaps(base_url: str) -> List[str]:
    root = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}"
    robots = fetch_xml(urljoin(root, "/robots.txt")) or ""
    urls = [line.split(":", 1)[1].strip() for line in robots.splitlines() if line.lower().startswith("sitemap:")]
    if not urls:
        urls = [urljoin(root, "/sitemap.xml")]
    return urls[:8]


def urls_from_sitemap(sitemap_url: str) -> List[str]:
    xml = fetch_xml(sitemap_url)
    if not xml:
        return []
    out: List[str] = []
    try:
        root = ET.fromstring(xml.encode("utf-8"))
    except Exception:
        return out
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    for loc in root.findall(".//sm:url/sm:loc", ns):
        if loc.text:
            out.append(loc.text.strip())
    for loc in root.findall(".//sm:sitemap/sm:loc", ns):
        if loc.text and ("news" in loc.text.lower() or "post" in loc.text.lower()):
            nested = fetch_xml(loc.text.strip())
            if not nested:
                continue
            try:
                nested_root = ET.fromstring(nested.encode("utf-8"))
            except Exception:
                continue
            for nloc in nested_root.findall(".//sm:url/sm:loc", ns):
                if nloc.text:
                    out.append(nloc.text.strip())
    return out


def discover_from_listing(url: str) -> List[str]:
    html = fetch_html(url)
    if not html:
        return []
    found: Set[str] = set()

    n = NEXT_DATA_RE.search(html)
    if n:
        try:
            data = json.loads(n.group(1))
            serialized = json.dumps(data)
            for m in re.findall(r'https?://[^"\'\s<>]+', serialized):
                found.add(m)
        except Exception:
            pass

    for href in A_HREF_RE.findall(html):
        abs_url = canonicalize_url(urljoin(url, href))
        if abs_url.startswith("http"):
            found.add(abs_url)

    return list(found)


def discover_non_rss_candidates(limit: int) -> List[Candidate]:
    out: List[Candidate] = []
    if limit <= 0:
        return out

    for source, listing in NON_RSS_LISTING_URLS.items():
        filtered: List[str] = []
        seen: Set[str] = set()

        for sm in discover_sitemaps(listing):
            for u in urls_from_sitemap(sm):
                cu = canonicalize_url(u)
                if not cu or cu in seen:
                    continue
                seen.add(cu)
                if is_story_url(cu):
                    filtered.append(cu)
                if len(filtered) >= MAX_NON_RSS_URLS_PER_SOURCE:
                    break
            if len(filtered) >= MAX_NON_RSS_URLS_PER_SOURCE:
                break

        if not filtered:
            for u in discover_from_listing(listing):
                cu = canonicalize_url(u)
                if not cu or cu in seen:
                    continue
                seen.add(cu)
                if is_story_url(cu):
                    filtered.append(cu)
                if len(filtered) >= MAX_NON_RSS_URLS_PER_SOURCE:
                    break

        log(f"non_rss_source={source} kept={len(filtered)} checked={len(seen)}")
        for u in filtered:
            out.append(Candidate(source=source, url=u))

    return out[:limit]


def enrich_and_validate(
    c: Candidate,
    state: Dict[str, Any],
    require_high_author: bool,
    rejected: Dict[str, int],
) -> Optional[Candidate]:
    resolved = resolve_final_url(c.url, state["redirect_cache"])
    domain = urlparse(resolved).netloc.lower()

    if is_bad_domain(domain):
        rejected["bad_domain"] += 1
        log(f"rejected_bad_domain {c.url} -> {resolved}")
        return None
    if not is_story_url(resolved):
        rejected["not_story_url"] += 1
        log(f"rejected_not_story_url {c.url} -> {resolved}")
        return None

    meta = state["meta_cache"].get(resolved)
    title = c.title.strip()
    author = c.author.strip()
    summary = clean_text(c.summary)
    image_url = c.image_url.strip()

    if meta:
        title = title or meta.get("title", "")
        author = author or meta.get("author", "")
        summary = summary or meta.get("summary", "")
        image_url = image_url or meta.get("image_url", "")
    else:
        html = fetch_html(resolved)
        if html:
            m_title, m_author, m_canonical, m_desc, m_image = extract_meta(resolved, html)
            if m_canonical:
                resolved = resolve_final_url(m_canonical, state["redirect_cache"])
                if not is_story_url(resolved):
                    rejected["not_story_url"] += 1
                    log(f"rejected_not_story_url canonical {m_canonical}")
                    return None
            title = title or m_title
            author = author or m_author
            summary = summary or m_desc
            image_url = image_url or m_image
        state["meta_cache"][resolved] = {
            "title": title,
            "author": author,
            "summary": summary,
            "image_url": image_url,
            "ts": now_utc().isoformat(),
        }

    if not giants_relevant(title, summary, c.categories, resolved):
        rejected["irrelevant"] += 1
        log(f"rejected_irrelevant {resolved}")
        return None

    c.url = resolved
    c.title = title or "Giants update"
    c.author = author
    c.summary = summary
    c.image_url = image_url
    return c


def validate_candidates(
    candidates: List[Candidate],
    state: Dict[str, Any],
    require_high_author: bool,
    rejected: Dict[str, int],
) -> List[Candidate]:
    out: List[Candidate] = []
    seen_urls: Set[str] = set()
    for c in candidates:
        if not c.url:
            continue
        if not is_recent_enough(c.published_ts):
            rejected["stale"] += 1
            continue

        normalized = canonicalize_url(c.url)
        if not normalized:
            continue
        if normalized in state["posted_urls"] or normalized in seen_urls:
            rejected["duplicate"] += 1
            continue

        # For non-rss candidates, if author already looks high-priority, skip fast prefilter.
        skip_prefilter = require_high_author and c.author and is_high_author(c.author)
        if not skip_prefilter and (c.title or c.summary or c.categories):
            if not giants_relevant(c.title, c.summary, c.categories, normalized):
                rejected["irrelevant_prefilter"] += 1
                continue

        c.url = normalized
        vc = enrich_and_validate(c, state, require_high_author=require_high_author, rejected=rejected)
        if vc and vc.url not in state["posted_urls"] and vc.url not in seen_urls:
            seen_urls.add(vc.url)
            out.append(vc)
    return out


def dedupe(candidates: List[Candidate], state: Dict[str, Any]) -> List[Candidate]:
    out: List[Candidate] = []
    seen_this_run: Set[str] = set()
    for c in candidates:
        if c.url in state["posted_urls"] or c.url in seen_this_run:
            continue
        seen_this_run.add(c.url)
        out.append(c)
    return out


def candidate_sort_key(c: Candidate, state: Dict[str, Any]) -> Tuple[int, float, str, str]:
    rank_group = 2 if c.discovered_via in {"google", "nonrss"} else 1
    dt = parse_dt_or_none(c.published_ts)
    if not dt:
        meta = state.get("meta_cache", {}).get(c.url, {})
        dt = parse_dt_or_none(meta.get("ts", "")) if isinstance(meta, dict) else None
    if not dt:
        dt = now_utc()
    return (rank_group, dt.timestamp(), c.source.lower(), c.url)


def select_with_source_cap(candidates: List[Candidate]) -> List[Candidate]:
    selected: List[Candidate] = []
    per_source: Dict[str, int] = defaultdict(int)

    for c in candidates:
        if len(selected) >= MAX_POSTS_PER_RUN:
            break
        if per_source[c.source] >= MAX_PER_SOURCE_PER_RUN:
            continue
        selected.append(c)
        per_source[c.source] += 1

    if len(selected) < MAX_POSTS_PER_RUN:
        selected_urls = {c.url for c in selected}
        for c in candidates:
            if len(selected) >= MAX_POSTS_PER_RUN:
                break
            if c.url in selected_urls:
                continue
            selected.append(c)
            selected_urls.add(c.url)

    return selected


def bsky_login() -> Tuple[str, str]:
    r = SESSION.post(
        f"{BSKY_PDS}/xrpc/com.atproto.server.createSession",
        json={"identifier": BSKY_IDENTIFIER, "password": BSKY_APP_PASSWORD},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    return data["did"], data["accessJwt"]


def truncate_line(line: str, max_len: int) -> str:
    if len(line) <= max_len:
        return line
    return line[: max_len - 1].rstrip() + "…"


def build_post_text(c: Candidate) -> str:
    first = f"{c.source}: {clean_text(c.title)}"
    first = truncate_line(first, 260)
    return f"{first}\n{c.url}"


def upload_external_thumb(image_url: str, jwt: str) -> Optional[Dict[str, Any]]:
    if not image_url:
        return None
    try:
        with SESSION.get(image_url, timeout=REQUEST_TIMEOUT, stream=True) as r:
            if r.status_code >= 400:
                return None
            content_type = r.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                return None
            blob_bytes = r.raw.read(900_000)
            if not blob_bytes:
                return None
    except Exception:
        return None

    try:
        upload = SESSION.post(
            f"{BSKY_PDS}/xrpc/com.atproto.repo.uploadBlob",
            headers={"Authorization": f"Bearer {jwt}", "Content-Type": content_type},
            data=blob_bytes,
            timeout=REQUEST_TIMEOUT,
        )
        upload.raise_for_status()
        return upload.json().get("blob")
    except Exception:
        return None


def create_embed_for_candidate(c: Candidate, jwt: str) -> Dict[str, Any]:
    description = truncate_line(clean_text(c.summary or ""), 280)
    if not description:
        description = c.source

    external: Dict[str, Any] = {
        "uri": c.url,
        "title": truncate_line(clean_text(c.title or "Giants update"), 100),
        "description": description,
    }

    thumb_blob = upload_external_thumb(c.image_url, jwt)
    if thumb_blob:
        external["thumb"] = thumb_blob

    return {"$type": "app.bsky.embed.external", "external": external}


def post_to_bluesky(c: Candidate, did: str, jwt: str) -> None:
    text = build_post_text(c)
    link_start = text.rfind(c.url)
    facets: List[Dict[str, Any]] = []
    if link_start >= 0:
        facets.append(
            {
                "index": {"byteStart": link_start, "byteEnd": link_start + len(c.url)},
                "features": [{"$type": "app.bsky.richtext.facet#link", "uri": c.url}],
            }
        )

    payload = {
        "repo": did,
        "collection": "app.bsky.feed.post",
        "record": {
            "$type": "app.bsky.feed.post",
            "text": text,
            "facets": facets,
            "embed": create_embed_for_candidate(c, jwt),
            "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    }
    r = SESSION.post(
        f"{BSKY_PDS}/xrpc/com.atproto.repo.createRecord",
        headers={"Authorization": f"Bearer {jwt}"},
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()


def main() -> None:
    state = state_load()
    prune_state(state)

    rejected: Dict[str, int] = defaultdict(int)

    rss_candidates = discover_from_rss()
    google_candidates = discover_from_google_news()

    rss_validated = validate_candidates(rss_candidates, state, require_high_author=False, rejected=rejected)
    google_validated = validate_candidates(google_candidates, state, require_high_author=True, rejected=rejected)

    nonrss_validated: List[Candidate] = []
    if len(rss_validated) + len(google_validated) < max(MAX_POSTS_PER_RUN, MAX_VALIDATION_TARGET):
        nonrss_limit = len(NON_RSS_LISTING_URLS) * MAX_NON_RSS_URLS_PER_SOURCE
        nonrss_candidates = discover_non_rss_candidates(limit=nonrss_limit)
        nonrss_validated = validate_candidates(nonrss_candidates, state, require_high_author=True, rejected=rejected)

    combined = dedupe(rss_validated + google_validated + nonrss_validated, state)
    combined.sort(key=lambda c: candidate_sort_key(c, state), reverse=True)
    selected = select_with_source_cap(combined)

    by_via = defaultdict(int)
    for c in combined:
        by_via[c.discovered_via or "unknown"] += 1
    log(f"validated_counts rss={by_via.get('rss', 0)} google={by_via.get('google', 0)} nonrss={by_via.get('nonrss', 0)} total={len(combined)}")
    log("rejected_counts " + " ".join(f"{k}={v}" for k, v in sorted(rejected.items())))

    for c in selected:
        log(f"selected source={c.source} via={c.discovered_via} author={c.author or '<none>'} title={c.title} url={c.url}")

    if not selected:
        log("No validated candidates.")
        state_save(state)
        return

    if DRY_RUN:
        for c in selected:
            log(f"DRY_RUN would post: {build_post_text(c)}")
            state["posted_urls"][c.url] = now_utc().isoformat()
        state_save(state)
        return

    if not BSKY_IDENTIFIER or not BSKY_APP_PASSWORD:
        raise RuntimeError("BSKY_IDENTIFIER and BSKY_APP_PASSWORD are required when not DRY_RUN")

    did, jwt = bsky_login()
    for c in selected:
        post_to_bluesky(c, did, jwt)
        state["posted_urls"][c.url] = now_utc().isoformat()
        log(f"posted {c.url}")
        time.sleep(0.8)

    state_save(state)


if __name__ == "__main__":
    main()
