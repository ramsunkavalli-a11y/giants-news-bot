#!/usr/bin/env python3
"""
SF Giants news -> Bluesky bot.

Configuration (env vars):
- BSKY_IDENTIFIER (required unless DRY_RUN=1)
- BSKY_APP_PASSWORD (required unless DRY_RUN=1)
- BSKY_PDS (default: https://bsky.social)
- STATE_FILE (default: state.json)
- MAX_POSTS_PER_RUN (default: 10)
- KEEP_POSTED_DAYS (default: 21)
- DRY_RUN (default: 0)
- USER_AGENT (optional)
- REQUEST_TIMEOUT (default: 15)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
import xml.etree.ElementTree as ET

import feedparser
import requests
from dateutil import parser as dtparser

UA = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 GiantsNewsBot/2026",
)
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))
DRY_RUN = os.getenv("DRY_RUN", "0").lower() in {"1", "true", "yes"}
STATE_FILE = os.getenv("STATE_FILE", "state.json")
MAX_POSTS_PER_RUN = int(os.getenv("MAX_POSTS_PER_RUN", "10"))
KEEP_POSTED_DAYS = int(os.getenv("KEEP_POSTED_DAYS", "21"))

BSKY_PDS = os.getenv("BSKY_PDS", "https://bsky.social")
BSKY_IDENTIFIER = os.getenv("BSKY_IDENTIFIER", "")
BSKY_APP_PASSWORD = os.getenv("BSKY_APP_PASSWORD", "")

RSS_SOURCES = [
    ("SF Standard", "https://sfstandard.com/category/sports/feed/"),
    ("SFGate Giants", "https://www.sfgate.com/giants/feed/Giants-447.php"),
    ("NYTimes Baseball", "https://rss.nytimes.com/services/xml/rss/nyt/Baseball.xml"),
]

NON_RSS_LISTING_URLS = {
    "NBC Sports Bay Area": "https://www.nbcsportsbayarea.com/mlb/san-francisco-giants/",
    "SF Chronicle Giants": "https://www.sfchronicle.com/sports/giants/",
    "Mercury News Giants": "https://www.mercurynews.com/tag/san-francisco-giants/",
    "AP Giants hub": "https://apnews.com/hub/san-francisco-giants",
    "MLB Giants News": "https://www.mlb.com/giants/news",
    "SFGiants News": "https://www.mlb.com/giants/news",  # same source of truth
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

META_TITLE_RE = re.compile(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', re.I)
META_AUTHOR_RE = re.compile(r'<meta[^>]+name=["\']author["\'][^>]+content=["\']([^"\']+)', re.I)
CANONICAL_RE = re.compile(r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)', re.I)
OG_URL_RE = re.compile(r'<meta[^>]+property=["\']og:url["\'][^>]+content=["\']([^"\']+)', re.I)
TITLE_TAG_RE = re.compile(r"<title>(.*?)</title>", re.I | re.S)
A_HREF_RE = re.compile(r"<a[^>]+href=[\"']([^\"']+)[\"']", re.I)
NEXT_DATA_RE = re.compile(r'<script[^>]*id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>', re.I | re.S)
JSON_LD_RE = re.compile(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.I | re.S)


@dataclass
class Candidate:
    source: str
    url: str
    title: str = ""
    author: str = ""
    summary: str = ""
    categories: List[str] = None

    def __post_init__(self) -> None:
        if self.categories is None:
            self.categories = []


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def log(msg: str) -> None:
    print(f"[{datetime.utcnow().isoformat()}] {msg}")


def canonicalize_url(url: Any) -> str:
    if not isinstance(url, str):
        return ""
    if not url:
        return ""
    url = url.strip()
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]

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
        r = requests.get(c, headers={"User-Agent": UA}, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        final = canonicalize_url(r.url or c)
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
            slug = last
            return "-" in slug and len(slug) >= 20 and slug not in STORY_REJECT_SLUGS

    if p.netloc.endswith("nbcsportsbayarea.com"):
        if last in {"giants-news", "giants-rumors", "giants-playoffs", "giants-spring-training"}:
            return False
        if len(segs) >= 3 and segs[-1].isdigit() and len(segs[-1]) >= 6 and "-" in segs[-2]:
            return True

    return "-" in last and len(last) >= 24 and last not in STORY_REJECT_SLUGS


def giants_relevant(title: str, summary: str, categories: Iterable[str], url: str) -> bool:
    text = " ".join([title or "", summary or "", " ".join(categories or []), url or ""]).lower()
    if any(term in text for term in NFL_TERMS):
        return False
    if "san francisco giants" in text or "sf giants" in text:
        return True
    if "giants" in text and any(term in text for term in BASEBALL_TERMS):
        return True
    return False


def fetch_html(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400 or "text/html" not in r.headers.get("Content-Type", ""):
            return None
        return r.text[:2_000_000]
    except Exception:
        return None


def extract_meta(url: str, html: str) -> Tuple[str, str, str]:
    title = ""
    author = ""
    canonical = ""

    m = META_TITLE_RE.search(html)
    if m:
        title = m.group(1).strip()
    if not title:
        t = TITLE_TAG_RE.search(html)
        if t:
            title = re.sub(r"\s+", " ", t.group(1)).strip()

    ma = META_AUTHOR_RE.search(html)
    if ma:
        author = ma.group(1).strip()

    c1 = CANONICAL_RE.search(html)
    c2 = OG_URL_RE.search(html)
    canonical = (c1.group(1) if c1 else "") or (c2.group(1) if c2 else "")
    canonical = canonicalize_url(urljoin(url, canonical)) if canonical else ""

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
            if not author and item.get("author"):
                a = item.get("author")
                if isinstance(a, dict):
                    author = str(a.get("name", "")).strip()
                elif isinstance(a, list) and a and isinstance(a[0], dict):
                    author = str(a[0].get("name", "")).strip()
        if title and author:
            break

    return title, author, canonical


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
        if isinstance(raw_val, dict):
            val = canonicalize_url(raw_val.get("final", ""))
        else:
            val = canonicalize_url(raw_val)
        normalized_redirect_cache[key] = val or key
    data["redirect_cache"] = normalized_redirect_cache

    return data


def state_save(state: Dict[str, Any]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def prune_state(state: Dict[str, Any]) -> None:
    cutoff = now_utc() - timedelta(days=KEEP_POSTED_DAYS)
    for url, ts in list(state["posted_urls"].items()):
        try:
            dt = dtparser.isoparse(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt < cutoff:
                del state["posted_urls"][url]
        except Exception:
            del state["posted_urls"][url]


def discover_from_rss() -> List[Candidate]:
    out: List[Candidate] = []
    for source_name, feed_url in RSS_SOURCES:
        feed = feedparser.parse(feed_url)
        for e in feed.entries[:60]:
            out.append(
                Candidate(
                    source=source_name,
                    url=getattr(e, "link", "") or "",
                    title=getattr(e, "title", "") or "",
                    author=getattr(e, "author", "") or "",
                    summary=getattr(e, "summary", "") or "",
                    categories=[t.get("term", "") for t in getattr(e, "tags", []) if isinstance(t, dict)],
                )
            )
    return out


def fetch_xml(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400:
            return None
        return r.text
    except Exception:
        return None


def discover_sitemaps(base_url: str) -> List[str]:
    root = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}"
    robots = fetch_xml(urljoin(root, "/robots.txt")) or ""
    sitemap_urls = []
    for line in robots.splitlines():
        if line.lower().startswith("sitemap:"):
            sitemap_urls.append(line.split(":", 1)[1].strip())
    if not sitemap_urls:
        sitemap_urls = [urljoin(root, "/sitemap.xml")]
    return sitemap_urls[:8]


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
                for nloc in nested_root.findall(".//sm:url/sm:loc", ns):
                    if nloc.text:
                        out.append(nloc.text.strip())
            except Exception:
                continue
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


def discover_non_rss_candidates() -> List[Candidate]:
    out: List[Candidate] = []
    for source, listing in NON_RSS_LISTING_URLS.items():
        urls: Set[str] = set()
        for sm in discover_sitemaps(listing):
            for u in urls_from_sitemap(sm):
                urls.add(u)
        if not urls:
            for u in discover_from_listing(listing):
                urls.add(u)
        for u in list(urls)[:200]:
            out.append(Candidate(source=source, url=u))
    return out


def enrich_and_validate(c: Candidate, state: Dict[str, Any]) -> Optional[Candidate]:
    resolved = resolve_final_url(c.url, state["redirect_cache"])
    domain = urlparse(resolved).netloc.lower()

    if is_bad_domain(domain):
        log(f"rejected_bad_domain {c.url} -> {resolved}")
        return None
    if not is_story_url(resolved):
        log(f"rejected_not_story_url {c.url} -> {resolved}")
        return None

    meta = state["meta_cache"].get(resolved)
    title = c.title.strip()
    author = c.author.strip()

    if meta:
        title = title or meta.get("title", "")
        author = author or meta.get("author", "")
    else:
        html = fetch_html(resolved)
        if html:
            m_title, m_author, m_canonical = extract_meta(resolved, html)
            if m_canonical:
                resolved = resolve_final_url(m_canonical, state["redirect_cache"])
                if not is_story_url(resolved):
                    log(f"rejected_not_story_url canonical {m_canonical}")
                    return None
            title = title or m_title
            author = author or m_author
        state["meta_cache"][resolved] = {"title": title, "author": author, "ts": now_utc().isoformat()}

    if not giants_relevant(title, c.summary, c.categories, resolved):
        log(f"rejected_irrelevant {resolved}")
        return None

    c.url = resolved
    c.title = title or "Giants update"
    c.author = author
    return c


def dedupe(candidates: List[Candidate], state: Dict[str, Any]) -> List[Candidate]:
    out: List[Candidate] = []
    seen_this_run: Set[str] = set()
    for c in candidates:
        if c.url in state["posted_urls"] or c.url in seen_this_run:
            log(f"rejected_duplicate {c.url}")
            continue
        seen_this_run.add(c.url)
        out.append(c)
    return out


def bsky_login() -> Tuple[str, str]:
    r = requests.post(
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
    who = c.author.strip() or c.source
    first = f"{who}: {c.title.strip()}"
    first = truncate_line(first, 260)
    return f"{first}\n{c.url}"


def post_to_bluesky(c: Candidate, did: str, jwt: str) -> None:
    text = build_post_text(c)
    payload = {
        "repo": did,
        "collection": "app.bsky.feed.post",
        "record": {
            "$type": "app.bsky.feed.post",
            "text": text,
            "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    }
    r = requests.post(
        f"{BSKY_PDS}/xrpc/com.atproto.repo.createRecord",
        headers={"Authorization": f"Bearer {jwt}"},
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()


def main() -> None:
    state = state_load()
    prune_state(state)

    candidates = discover_from_rss() + discover_non_rss_candidates()
    validated: List[Candidate] = []
    for c in candidates:
        if not c.url:
            continue
        vc = enrich_and_validate(c, state)
        if vc:
            validated.append(vc)

    validated = dedupe(validated, state)
    validated = validated[:MAX_POSTS_PER_RUN]

    if not validated:
        log("No validated candidates.")
        state_save(state)
        return

    if DRY_RUN:
        for c in validated:
            log(f"DRY_RUN would post: {build_post_text(c)}")
            state["posted_urls"][c.url] = now_utc().isoformat()
        state_save(state)
        return

    if not BSKY_IDENTIFIER or not BSKY_APP_PASSWORD:
        raise RuntimeError("BSKY_IDENTIFIER and BSKY_APP_PASSWORD are required when not DRY_RUN")

    did, jwt = bsky_login()
    for c in validated:
        post_to_bluesky(c, did, jwt)
        state["posted_urls"][c.url] = now_utc().isoformat()
        log(f"posted {c.url}")
        time.sleep(0.8)

    state_save(state)


if __name__ == "__main__":
    main()


# Inline test examples for is_story_url (expected):
# - https://www.nbcsportsbayarea.com/mlb/san-francisco-giants/giants-news/ -> False
# - https://www.nbcsportsbayarea.com/mlb/san-francisco-giants/buster-posey-update/1749203/ -> True
# - https://apnews.com/hub/san-francisco-giants -> False
# - https://apnews.com/article/san-francisco-giants-abc123def456 -> True
# - https://www.mlb.com/giants/news/giants-call-up-top-prospect-for-debut -> True
# - https://www.mlb.com/giants/news -> False
