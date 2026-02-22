import json
import argparse
import base64
import html as html_lib
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse, urlencode, urlunparse

import feedparser
import requests
from dateutil import parser as dtparser


# -----------------------------
# Config
# -----------------------------
TEAM_ID = 137  # SF Giants (MLB Stats API)

HOURS_BACK = int(os.getenv("HOURS_BACK", "8"))
# 0 => unlimited
MAX_POSTS_PER_RUN = int(os.getenv("MAX_POSTS_PER_RUN", "0") or "0")
STATE_FILE = os.getenv("STATE_FILE", "state.json")

# Per-run diversity is off by default. Enable explicitly if you want it.
ENFORCE_PER_SOURCE_CAP = os.getenv("ENFORCE_PER_SOURCE_CAP", "false").lower() in {"1", "true", "yes"}
PER_SOURCE_CAP = int(os.getenv("PER_SOURCE_CAP", "0") or "0")

# Non-primary daily cap
OTHER_DAILY_CAP = int(os.getenv("OTHER_DAILY_CAP", "2"))

# Cache lifetimes
ROSTER_CACHE_HOURS = int(os.getenv("ROSTER_CACHE_HOURS", "24"))
STAFF_CACHE_HOURS = int(os.getenv("STAFF_CACHE_HOURS", "24"))
KEEP_POSTED_DAYS = int(os.getenv("KEEP_POSTED_DAYS", "21"))

DEBUG_REJECTIONS = os.getenv("DEBUG_REJECTIONS", "0") == "1"

# Google News RSS can be large; resolving each item involves a fetch.
MAX_GOOGLE_ENTRIES_PER_FEED = int(os.getenv("MAX_GOOGLE_ENTRIES_PER_FEED", "25"))

# Listing pages can have tons of links; cap candidate processing.
MAX_LISTING_LINKS_PER_SOURCE = int(os.getenv("MAX_LISTING_LINKS_PER_SOURCE", "30"))

# External embed makes a “card” appear like pasting a URL.
ENABLE_EXTERNAL_EMBED = os.getenv("ENABLE_EXTERNAL_EMBED", "true").lower() in {"1", "true", "yes"}

# Metadata enrichment for embed title/description/author; costs extra fetches.
ENRICH_ARTICLE_METADATA = os.getenv("ENRICH_ARTICLE_METADATA", "true").lower() in {"1", "true", "yes"}
MAX_META_FETCHES_PER_RUN = int(os.getenv("MAX_META_FETCHES_PER_RUN", "25"))

# Listing pages are our non-RSS path. Keep this separate from embed enrichment settings.
ENABLE_LISTING_FETCH = os.getenv("ENABLE_LISTING_FETCH", "true").lower() in {"1", "true", "yes"}
MAX_LISTING_META_FETCHES_PER_RUN = int(os.getenv("MAX_LISTING_META_FETCHES_PER_RUN", "40"))
MAX_LISTING_META_FETCHES_PER_SOURCE = int(os.getenv("MAX_LISTING_META_FETCHES_PER_SOURCE", "10"))
SITEMAP_CACHE_HOURS = int(os.getenv("SITEMAP_CACHE_HOURS", "6"))
REDIRECT_CACHE_HOURS = int(os.getenv("REDIRECT_CACHE_HOURS", "24"))
DRY_RUN = os.getenv("DRY_RUN", "0") in {"1", "true", "yes"}
GOOGLE_ALERTS_RSS = [x.strip() for x in os.getenv("GOOGLE_ALERTS_RSS", "").split(",") if x.strip()]

# UA: use a browser-ish UA; some paywalled sites block bot UAs.
UA = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0 Safari/537.36 GiantsNewsBot/3.0",
)

# Paywall tagging
PAYWALL_DOMAINS = {
    "theathletic.com",
    "mercurynews.com",
    "baseballamerica.com",
    "sfchronicle.com",
}

BSKY_IDENTIFIER = os.environ["BSKY_IDENTIFIER"]
BSKY_APP_PASSWORD = os.environ["BSKY_APP_PASSWORD"]
BSKY_PDS = os.getenv("BSKY_PDS", "https://bsky.social")


# -----------------------------
# Domains you explicitly care about
# -----------------------------
PRIMARY_DOMAINS = {
    "sfchronicle.com",
    "mercurynews.com",
    "nbcsportsbayarea.com",
    "sfgiants.com",
    "mlb.com",
    "sfstandard.com",
    "knbr.com",
    "sfgate.com",
    "theathletic.com",
    "apnews.com",
    "fangraphs.com",
    "baseballamerica.com",
}



SOURCE_EXPECTED_DOMAINS: Dict[str, Set[str]] = {
    # Keep strict domain checks only where we repeatedly saw wrong targets.
    "Google News: Mercury News": {"mercurynews.com"},
    "Google News: SFGiants.com / MLB Giants": {"sfgiants.com", "mlb.com"},
}
AGGREGATOR_BLOCKLIST = {
    "news.google.com",
    "feedspot.com",
    "feedly.com",
    "newsbreak.com",
    "ground.news",
}

# Tracker / junk domains we never want to post as the “article URL”
TRACKER_BLOCKLIST = {
    "google-analytics.com",
    "www.google-analytics.com",
    "doubleclick.net",
    "googlesyndication.com",
    "adsystem.com",
    "adservice.google.com",
    "securepubads.g.doubleclick.net",
    "tpc.googlesyndication.com",
    "stats.g.doubleclick.net",
    "ad.doubleclick.net",
}

# Metadata/reference domains sometimes appear in scraped html but are never article URLs.
REFERENCE_DOMAIN_BLOCKLIST = {
    "w3.org",
    "www.w3.org",
    "schema.org",
    "www.schema.org",
    "fonts.googleapis.com",
}


# -----------------------------
# Always-relevant names
# -----------------------------
FRONT_OFFICE_POWER = {
    "Greg Johnson",
    "Rob Dean",
    "Larry Baer",
    "Buster Posey",
    "Zack Minasian",
    "Jeremy Shelley",
    "Paul Bien",
    "Randy Winn",
}
KEY_PEOPLE = set(FRONT_OFFICE_POWER) | {"Tony Vitello"}


# -----------------------------
# Relevance rules
# -----------------------------
BASEBALL_CONTEXT_TERMS = [
    "mlb", "major league", "baseball", "spring training", "cactus league", "grapefruit league",
    "opening day", "postseason", "playoffs", "world series", "nl west", "national league",
    "trade", "traded", "acquired", "deal", "deadline",
    "dfa", "designated for assignment", "waivers", "claimed",
    "optioned", "option", "call-up", "called up", "sent down",
    "roster", "40-man", "40 man", "injured list", "il", "rehab assignment",
    "pitcher", "starter", "rotation", "bullpen", "reliever", "closer",
    "catcher", "shortstop", "second base", "third base", "outfield", "first base", "dh",
    "inning", "innings", "era", "fip", "whip", "strikeout", "strikeouts", "walks", "ks",
    "home run", "homer", "batting", "slugging", "ops", "wrc+", "war", "xwoba",
    "prospect", "prospects", "farm system", "scouting", "draft", "international signing",
    "player development", "minor league", "triple-a", "double-a",
    "manager", "bench coach", "pitching coach", "hitting coach", "coach",
    "general manager", "gm", "front office", "president of baseball operations",
]

NEGATIVE_PHRASES = [
    "dear abby",
    "new york giants", "ny giants",
    "nfl", "super bowl", "touchdown", "quarterback", "wide receiver", "linebacker",
]

BASEBALL_SOURCE_HINTS = [
    "fangraphs", "baseball america", "mlb", "sfgiants", "baseball prospectus",
]

IMPORTANT_KEYWORDS = {
    "trade", "traded", "acquire", "acquired", "deal", "waiver", "waivers",
    "claimed", "dfa", "designated for assignment", "optioned", "call-up", "called up",
    "sign", "signed", "signing", "extension",
    "injury", "injured", "il", "injured list", "surgery", "rehab",
    "prospect", "prospects", "promotion", "promoted",
    "rotation", "bullpen", "starter", "closer",
}


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Item:
    title: str
    url: str
    publication: str
    published: datetime
    domain: str
    is_primary: bool
    author: str = ""
    score: int = 0
    raw_summary: str = ""


# -----------------------------
# Regex helpers
# -----------------------------
RE_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
RE_SPACE = re.compile(r"\s+")
RE_URL = re.compile(r"https?://[^\s\"\'<>]+", re.I)

def looks_like_url(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return bool(RE_URL.fullmatch(t))


# -----------------------------
# Time / text helpers
# -----------------------------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def norm_text(s: str) -> str:
    return RE_SPACE.sub(" ", (s or "").strip().lower())


def tokenize_words(text: str) -> Set[str]:
    return set(w.lower() for w in RE_WORD.findall(text or ""))


def safe_get(url: str, timeout: int = 25, retries: int = 2, backoff: float = 0.8) -> requests.Response:
    last_err: Optional[Exception] = None
    for i in range(max(1, retries + 1)):
        try:
            return requests.get(url, timeout=timeout, headers={"User-Agent": UA}, allow_redirects=True)
        except Exception as ex:
            last_err = ex
            if i >= retries:
                break
            time.sleep(backoff * (2 ** i))
    raise last_err if last_err else RuntimeError("safe_get failed")


def _jina_proxy_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    return f"https://r.jina.ai/{u}"


def fetch_text_via_jina(url: str, timeout: int = 25) -> str:
    """
    Fallback fetch through r.jina.ai to bypass bot/proxy blocks on publisher pages.
    Returns plain text/markdown-like content or empty string.
    """
    proxy_url = _jina_proxy_url(url)
    if not proxy_url:
        return ""
    try:
        r = requests.get(proxy_url, timeout=timeout, headers={"User-Agent": UA}, allow_redirects=True)
        if r.status_code >= 400:
            return ""
        return r.text or ""
    except Exception:
        return ""


def domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def is_google_host(url: str) -> bool:
    host = domain_of(url)
    return (
        "news.google.com" in host
        or host.endswith("google.com")
        or host.endswith("googleusercontent.com")
        or host.endswith("gstatic.com")
    )


def is_blocked_domain(domain: str) -> bool:
    d = (domain or "").lower()
    if not d:
        return True
    if d in TRACKER_BLOCKLIST or any(d.endswith("." + x) for x in TRACKER_BLOCKLIST):
        return True
    if d in AGGREGATOR_BLOCKLIST or any(d.endswith("." + x) for x in AGGREGATOR_BLOCKLIST):
        return True
    if d.endswith("google.com") or d.endswith("googleusercontent.com") or d.endswith("gstatic.com") or d.endswith("googleapis.com"):
        return True
    return False


def is_reference_domain(domain: str) -> bool:
    d = (domain or "").lower()
    if not d:
        return True
    return d in REFERENCE_DOMAIN_BLOCKLIST or any(d.endswith("." + x) for x in REFERENCE_DOMAIN_BLOCKLIST)


def expected_domains_for_source(source_label: str) -> Set[str]:
    return SOURCE_EXPECTED_DOMAINS.get(source_label, set())


def is_home_or_section_root_url(url: str) -> bool:
    """
    Reject generic home/section roots that are not actual story pages.
    """
    try:
        path = (urlparse(url).path or "").strip("/").lower()
    except Exception:
        return True
    if not path:
        return True

    rootish = {
        "news", "giants", "mlb", "sports", "team", "teams", "baseball",
        "giants/news", "mlb/team/giants", "athletic/mlb/team/giants",
    }
    if path in rootish:
        return True

    # very shallow section path
    segs = [x for x in path.split("/") if x]
    if len(segs) <= 1:
        return True
    return False


def is_likely_story_url(url: str) -> bool:
    try:
        path = (urlparse(url).path or "").strip("/").lower()
    except Exception:
        return False
    if not path:
        return False
    if re.search(r"/20\d{2}/\d{1,2}/\d{1,2}/", "/" + path):
        return True
    segs = [x for x in path.split("/") if x]
    if len(segs) >= 3 and any("-" in seg for seg in segs):
        return True
    if path.startswith("giants/news/") and len(segs) >= 3:
        return True
    if path.startswith("news/") and len(segs) >= 2 and "-" in segs[-1]:
        return True
    return False


def is_mlb_utility_or_evergreen(url: str, title: str) -> bool:
    """
    Reject known utility/evergreen MLB Giants pages that are not actual news stories.
    """
    t = norm_text(title)
    pth = (urlparse(url).path or "").lower()
    bad_terms = (
        "all-time lists", "trivia", "tv stream", "schedule", "tickets", "roster",
        "depth chart", "spring training tickets", "stats", "standings", "press releases",
        "injury report",
    )
    bad_path_terms = (
        "/schedule", "/tickets", "/roster", "/stats", "/standings", "/video",
        "/topic/", "/topics/", "/tag/", "/tags/",
        "all-time-lists", "trivia", "tv-stream", "press-releases", "injury-report",
    )
    if any(term in t for term in bad_terms):
        return True
    if any(term in pth for term in bad_path_terms):
        return True
    return False


def is_strict_story_url_for_source(source_label: str, url: str, title: str) -> bool:
    """
    Source-specific stricter checks for noisy feeds.
    """
    path = (urlparse(url).path or "").strip("/").lower()
    segs = [x for x in path.split("/") if x]

    if source_label in {"Google News: SFGiants.com / MLB Giants", "MLB Giants News", "SFGiants.com News"}:
        # Require article paths directly under /news or /giants/news and reject topic/tag sections.
        if not (path.startswith("giants/news/") or path.startswith("news/")):
            return False
        if any(x in {"topic", "topics", "tag", "tags"} for x in segs):
            return False

        # Expect exactly one slug after .../news/
        if path.startswith("giants/news/"):
            tail = path[len("giants/news/"):]
        else:
            tail = path[len("news/"):]
        if not tail or "/" in tail:
            return False
        if "-" not in tail:
            return False
        if is_mlb_utility_or_evergreen(url, title):
            return False

    return True


def canonicalize_url(url: str) -> str:
    """
    Remove obvious tracking params, fragments, normalize.
    This improves dedupe and reduces Google/UTM noise.
    """
    u = (url or "").strip()
    if not u:
        return u

    # Common extraction artifact from markdown/text mirrors.
    while u and u[-1] in ")],.;":
        u = u[:-1]

    try:
        p = urlparse(u)
        q = parse_qs(p.query, keep_blank_values=True)

        # Drop tracking keys
        drop_prefixes = ("utm_",)
        drop_keys = {
            "fbclid", "gclid", "mc_cid", "mc_eid", "cmpid", "ref", "refsrc", "source",
            "smid", "spm", "igshid", "mkt_tok"
        }
        new_q = {}
        for k, v in q.items():
            lk = k.lower()
            if lk in drop_keys:
                continue
            if any(lk.startswith(pref) for pref in drop_prefixes):
                continue
            new_q[k] = v

        query = urlencode(new_q, doseq=True)
        p2 = p._replace(query=query, fragment="")
        return urlunparse(p2)
    except Exception:
        return u


def title_from_url(url: str) -> str:
    """
    Fallback title when article metadata is unavailable.
    """
    try:
        path = (urlparse(url).path or "").strip("/")
        parts = [seg for seg in path.split("/") if seg]

        # Prefer a slug-like segment over numeric ID/date segments.
        slug = ""
        for seg in reversed(parts):
            seg_l = seg.lower()
            if re.fullmatch(r"\d+", seg_l):
                continue
            if re.fullmatch(r"20\d{2}", seg_l):
                continue
            if re.fullmatch(r"\d{1,2}", seg_l):
                continue
            slug = seg
            break
        if not slug and parts:
            slug = parts[-1]

        slug = slug.replace(")", "").replace("(", "")
        slug = re.sub(r"\.[A-Za-z0-9]+$", "", slug)
        slug = re.sub(r"[-_]+", " ", slug)
        slug = re.sub(r"\b\d{6,}\b", "", slug)
        slug = RE_SPACE.sub(" ", slug).strip()
        slug = re.sub(r"\ba s\b", "as", slug, flags=re.I)
        if not slug:
            return url
        return slug[:1].upper() + slug[1:]
    except Exception:
        return url


def date_from_url(url: str) -> Optional[datetime]:
    """
    Best-effort publish date extraction from common URL patterns like /2026/02/21/.
    """
    try:
        m = re.search(r"/(20\d{2})/(\d{1,2})/(\d{1,2})(?:/|$)", urlparse(url).path or "")
        if not m:
            return None
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return datetime(y, mo, d, tzinfo=timezone.utc)
    except Exception:
        return None


def is_paywalled_domain(domain: str) -> bool:
    d = (domain or "").lower()
    return any(d == pw or d.endswith("." + pw) for pw in PAYWALL_DOMAINS)


# -----------------------------
# State
# -----------------------------
def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {
            "posted": {},
            "roster_cache": {},
            "staff_cache": {},
            "daily_other": {},
            "last_run_success_at": None,
            "redirect_cache": {},
            "sitemap_cache": {},
            "source_seen_urls": {},
        }
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)
    state.setdefault("posted", {})
    state.setdefault("roster_cache", {})
    state.setdefault("staff_cache", {})
    state.setdefault("daily_other", {})
    state.setdefault("last_run_success_at", None)
    state.setdefault("redirect_cache", {})
    state.setdefault("sitemap_cache", {})
    state.setdefault("source_seen_urls", {})
    return state


def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def prune_state(state: Dict[str, Any], keep_days: int = KEEP_POSTED_DAYS) -> None:
    cutoff = utcnow() - timedelta(days=keep_days)
    posted = state.get("posted", {})
    to_del = []
    for url, ts in posted.items():
        try:
            dt = dtparser.isoparse(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt < cutoff:
                to_del.append(url)
        except Exception:
            to_del.append(url)
    for k in to_del:
        posted.pop(k, None)
    state["posted"] = posted

    # prune redirect cache
    rc = state.get("redirect_cache", {})
    rc_cutoff = utcnow() - timedelta(hours=REDIRECT_CACHE_HOURS)
    for k, row in list(rc.items()):
        try:
            ts = dtparser.isoparse((row or {}).get("ts", ""))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < rc_cutoff:
                rc.pop(k, None)
        except Exception:
            rc.pop(k, None)
    state["redirect_cache"] = rc


def get_daily_other_counter(state: Dict[str, Any]) -> Dict[str, Any]:
    today = utcnow().date().isoformat()
    daily = state.get("daily_other", {"date": today, "count": 0})
    if daily.get("date") != today:
        daily = {"date": today, "count": 0}
    state["daily_other"] = daily
    return daily


def compute_cutoff(state: Dict[str, Any]) -> datetime:
    base = utcnow() - timedelta(hours=HOURS_BACK)
    last_success_raw = state.get("last_run_success_at")
    if not last_success_raw:
        return base
    try:
        last_success = dtparser.isoparse(last_success_raw)
        if last_success.tzinfo is None:
            last_success = last_success.replace(tzinfo=timezone.utc)
        buffered = last_success - timedelta(minutes=15)
        return min(base, buffered)
    except Exception:
        return base


# -----------------------------
# MLB Stats API: roster + coaches
# -----------------------------
def load_cached_names(state: Dict[str, Any], key: str, max_age_hours: int) -> Set[str]:
    cache = state.get(key, {})
    ts = cache.get("fetched_at")
    if ts:
        try:
            fetched = dtparser.isoparse(ts)
            if fetched.tzinfo is None:
                fetched = fetched.replace(tzinfo=timezone.utc)
            if utcnow() - fetched < timedelta(hours=max_age_hours):
                return set(cache.get("names", []))
        except Exception:
            pass
    return set()


def save_cached_names(state: Dict[str, Any], key: str, names: Set[str]) -> None:
    state[key] = {"fetched_at": utcnow().isoformat(), "names": sorted(names)}


def statsapi_40man_names() -> Set[str]:
    url = f"https://statsapi.mlb.com/api/v1/teams/{TEAM_ID}/roster/40Man"
    r = safe_get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    names: Set[str] = set()
    for row in data.get("roster", []):
        person = row.get("person", {}) or {}
        full = (person.get("fullName") or "").strip()
        if full:
            names.add(full)
    return names


def statsapi_coaches_names() -> Set[str]:
    year = utcnow().year
    candidates = [
        f"https://statsapi.mlb.com/api/v1/teams/{TEAM_ID}/coaches?season={year}",
        f"https://statsapi.mlb.com/api/v1/teams/{TEAM_ID}/coaches",
    ]
    last_err = None
    for url in candidates:
        try:
            r = safe_get(url, timeout=20)
            r.raise_for_status()
            data = r.json()
            coaches = data.get("coaches") or data.get("teamCoaches") or data.get("roster") or []
            names: Set[str] = set()
            for row in coaches:
                person = row.get("person") or {}
                full = (person.get("fullName") or row.get("fullName") or row.get("name") or "").strip()
                if full:
                    names.add(full)
            if names:
                return names
        except Exception as e:
            last_err = e
            continue

    if last_err:
        print(f"[warn] coaches fetch not available: {last_err}")
    return set()


# -----------------------------
# Name matching
# -----------------------------
def last_name_token(full_name: str) -> Optional[str]:
    parts = [p for p in RE_WORD.findall(full_name or "") if p]
    return parts[-1].lower() if parts else None


def build_name_matchers(names: Set[str]) -> Tuple[Set[str], Dict[str, Set[str]]]:
    full_set = set(n.lower() for n in names if n)
    last_map: Dict[str, Set[str]] = {}
    for n in full_set:
        ln = last_name_token(n)
        if ln:
            last_map.setdefault(ln, set()).add(n)
    return full_set, last_map


def mentions_full_name(text: str, full_names: Set[str]) -> bool:
    t = norm_text(text)
    return any(n in t for n in full_names)


def contains_phrase(text: str, phrase: str) -> bool:
    t = norm_text(text)
    p = norm_text(phrase)
    return re.search(rf"(?<!\w){re.escape(p)}(?!\w)", t) is not None


def mentions_last_name_tight(text: str, last_name_map: Dict[str, Set[str]]) -> bool:
    t = norm_text(text)
    anchors_ok = (
        contains_phrase(t, "giants")
        or "san francisco" in t
        or contains_phrase(t, "sf")
        or "mlb" in t
        or "baseball" in t
        or "oracle park" in t
    )
    if not anchors_ok:
        return False
    words = tokenize_words(text)
    return any(ln in words for ln in last_name_map.keys())


# -----------------------------
# Relevance + scoring
# -----------------------------
def has_negative(text: str) -> bool:
    t = norm_text(text)
    return any(neg in t for neg in NEGATIVE_PHRASES)


def has_baseball_context(text: str, source_label: str) -> bool:
    t = norm_text(text)
    if any(term in t for term in BASEBALL_CONTEXT_TERMS):
        return True
    sl = norm_text(source_label)
    return any(h in sl for h in BASEBALL_SOURCE_HINTS)


def mentions_team_strong(text: str) -> bool:
    t = norm_text(text)
    return ("san francisco giants" in t) or ("sf giants" in t)


def is_allowed_item(
    title: str,
    summary: str,
    publication: str,
    domain: str,
    full_names: Set[str],
    last_map: Dict[str, Set[str]],
) -> Tuple[bool, str]:
    blob = f"{title}\n{summary}"
    t = norm_text(blob)

    if has_negative(blob):
        return False, "negative_phrase"

    # Recall boost: primary domains with "Giants" mention can pass even when summary is thin.
    is_primary_domain = any(domain == d or domain.endswith("." + d) for d in PRIMARY_DOMAINS)
    if is_primary_domain and contains_phrase(t, "giants"):
        return True, "primary_giants"

    if not has_baseball_context(blob, publication):
        return False, "no_baseball_context"

    if mentions_team_strong(blob):
        return True, "team_match"

    if full_names and (mentions_full_name(blob, full_names) or mentions_last_name_tight(blob, last_map)):
        return True, "name_match"

    return False, "no_team_or_name_match"


def importance_score(
    title: str,
    summary: str,
    publication: str,
    full_names: Set[str],
    key_people: Set[str],
    published: datetime,
    echo_count: int,
) -> int:
    blob = f"{title} {summary}".lower()
    score = 0

    for name in key_people:
        if name.lower() in blob:
            score += 10

    if any(n in blob for n in full_names):
        score += 6

    if ("san francisco giants" in blob) or ("sf giants" in blob):
        score += 4

    for kw in IMPORTANT_KEYWORDS:
        if kw in blob:
            score += 2

    if echo_count >= 2:
        score += min(8, 3 * (echo_count - 1))

    hours_old = max(0.0, (utcnow() - published).total_seconds() / 3600.0)
    if hours_old <= 2:
        score += 3
    elif hours_old <= 6:
        score += 2
    elif hours_old <= 12:
        score += 1

    pl = (publication or "").lower()
    if any(h in pl for h in BASEBALL_SOURCE_HINTS):
        score += 1

    return score


# -----------------------------
# Google News RSS URL
# -----------------------------
def google_news_rss_url(query: str) -> str:
    q = quote(query)
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def looks_like_google_news_rss(feed_url: str) -> bool:
    u = (feed_url or "").lower()
    return "news.google.com/rss/" in u or "news.google.com/rss" in u


def decode_google_redirect_url(url: str) -> str:
    """
    Handles cases like https://www.google.com/url?...&url=<target>
    """
    u = (url or "").strip()
    if not u:
        return u
    try:
        parsed = urlparse(u)
        host = parsed.netloc.lower()
        if host.endswith("google.com") and parsed.path.startswith("/url"):
            q = parse_qs(parsed.query)
            for key in ("url", "q", "u"):
                vals = q.get(key) or []
                for val in vals:
                    cand = unquote(val).strip()
                    if cand.startswith("http") and not is_google_host(cand):
                        return cand
    except Exception:
        pass
    return u


# -----------------------------
# HTML parsing (meta, canonical, json-ld, links)
# -----------------------------
class MetaLinkParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.links: List[str] = []
        self.canonical: str = ""
        self.og_url: str = ""
        self.title: str = ""
        self.og_title: str = ""
        self.description: str = ""
        self.og_description: str = ""
        self.author: str = ""
        self.published: str = ""
        self._in_title = False
        self._jsonld_buf: List[str] = []
        self._in_jsonld = False
        self.jsonld_blobs: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        attr = {k.lower(): (v or "") for k, v in attrs}

        if tag.lower() == "a":
            href = attr.get("href", "").strip()
            if href:
                self.links.append(urljoin(self.base_url, href))

        if tag.lower() == "link" and attr.get("rel", "").lower() == "canonical":
            href = attr.get("href", "").strip()
            if href:
                self.canonical = urljoin(self.base_url, href)

        if tag.lower() == "meta":
            prop = attr.get("property", "").lower()
            name = attr.get("name", "").lower()
            content = (attr.get("content") or "").strip()

            if prop == "og:url" and content:
                self.og_url = urljoin(self.base_url, content)
            if prop == "og:title" and content:
                self.og_title = content
            if prop == "og:description" and content:
                self.og_description = content

            # Common author fields
            if name in {"author", "parsely-author", "article:author", "byl"} and content:
                self.author = content

            # Common published time fields
            if prop in {"article:published_time", "og:updated_time"} and content:
                self.published = content
            if name in {"pubdate", "publishdate", "date", "dc.date", "dc.date.issued"} and content and not self.published:
                self.published = content

            # Description
            if name == "description" and content and not self.description:
                self.description = content

        if tag.lower() == "title":
            self._in_title = True

        if tag.lower() == "script":
            t = attr.get("type", "").lower()
            if t == "application/ld+json":
                self._in_jsonld = True
                self._jsonld_buf = []

    def handle_endtag(self, tag: str):
        if tag.lower() == "title":
            self._in_title = False
        if tag.lower() == "script" and self._in_jsonld:
            blob = "".join(self._jsonld_buf).strip()
            if blob:
                self.jsonld_blobs.append(blob)
            self._in_jsonld = False
            self._jsonld_buf = []

    def handle_data(self, data: str):
        if self._in_title:
            self.title += data
        if self._in_jsonld:
            self._jsonld_buf.append(data)


def _extract_from_jsonld(blobs: List[str]) -> Dict[str, str]:
    """
    Best-effort extraction of title/author/datePublished from JSON-LD.
    We avoid importing heavy libs; just json + shape checks.
    """
    out = {"title": "", "author": "", "published": "", "description": "", "canonical": ""}
    for b in blobs:
        try:
            obj = json.loads(b)
        except Exception:
            continue

        candidates = []
        if isinstance(obj, list):
            candidates = obj
        elif isinstance(obj, dict):
            # Some sites nest under @graph
            if isinstance(obj.get("@graph"), list):
                candidates = obj["@graph"]
            else:
                candidates = [obj]
        else:
            continue

        for c in candidates:
            if not isinstance(c, dict):
                continue
            t = str(c.get("@type") or "").lower()
            if t not in {"newsarticle", "article", "reportage", "blogposting"} and "article" not in t:
                continue

            headline = c.get("headline") or c.get("name") or ""
            if headline and not out["title"]:
                out["title"] = str(headline).strip()

            date_p = c.get("datePublished") or c.get("dateCreated") or ""
            if date_p and not out["published"]:
                out["published"] = str(date_p).strip()

            desc = c.get("description") or ""
            if desc and not out["description"]:
                out["description"] = str(desc).strip()

            author = c.get("author")
            if author and not out["author"]:
                if isinstance(author, dict):
                    out["author"] = str(author.get("name") or "").strip()
                elif isinstance(author, list):
                    # take first
                    a0 = author[0]
                    if isinstance(a0, dict):
                        out["author"] = str(a0.get("name") or "").strip()
                    else:
                        out["author"] = str(a0).strip()
                else:
                    out["author"] = str(author).strip()

            main_entity = c.get("mainEntityOfPage")
            if main_entity and not out["canonical"]:
                if isinstance(main_entity, dict):
                    out["canonical"] = str(main_entity.get("@id") or main_entity.get("url") or "").strip()
                else:
                    out["canonical"] = str(main_entity).strip()

        # If we got something useful, stop early
        if out["title"] or out["author"] or out["published"] or out["canonical"]:
            return out
    return out


def parse_html_metadata(url: str, html: str) -> Dict[str, str]:
    p = MetaLinkParser(base_url=url)
    try:
        p.feed(html or "")
    except Exception:
        pass

    j = _extract_from_jsonld(p.jsonld_blobs)

    canonical = p.canonical or p.og_url or j.get("canonical") or ""
    title = (p.og_title or j.get("title") or p.title or "").strip()
    author = (p.author or j.get("author") or "").strip()
    published = (p.published or j.get("published") or "").strip()
    description = (p.og_description or j.get("description") or p.description or "").strip()

    return {
        "canonical": canonical,
        "title": title,
        "author": author,
        "published": published,
        "description": description,
    }


def extract_links_from_listing(url: str, html: str) -> List[str]:
    p = MetaLinkParser(base_url=url)
    try:
        p.feed(html or "")
    except Exception:
        pass
    # Return unique, preserving order
    seen = set()
    out = []
    for u in p.links:
        u2 = canonicalize_url(u)
        if u2 not in seen:
            seen.add(u2)
            out.append(u2)
    return out


def extract_urls_from_text_blob(text: str) -> List[str]:
    """
    Extract URLs from plain text/markdown content (used by fallback fetchers).
    """
    seen = set()
    out: List[str] = []
    for u in RE_URL.findall(text or ""):
        u2 = canonicalize_url(u.strip())
        if u2 and u2 not in seen:
            seen.add(u2)
            out.append(u2)
    return out


def fallback_title_from_text(text: str, url: str) -> str:
    lines = (text or "").splitlines()

    # Prefer explicit "Title:" line if present in mirror output.
    for line in lines:
        t = line.strip()
        if t.lower().startswith("title:"):
            cand = t.split(":", 1)[1].strip()
            if cand:
                return cand[:300]

    skip_prefixes = (
        "url source:",
        "published time:",
        "markdown content:",
        "warning:",
    )
    for line in lines:
        t = line.strip().lstrip("#").strip()
        if len(t) < 12:
            continue
        tl = t.lower()
        if tl.startswith(skip_prefixes):
            continue
        if t.startswith("http://") or t.startswith("https://"):
            continue
        return t[:300]
    return title_from_url(url)


def fallback_description_from_text(text: str, max_len: int = 240) -> str:
    skip_prefixes = (
        "url source:",
        "published time:",
        "markdown content:",
        "title:",
        "warning:",
    )
    kept = []
    for line in (text or "").splitlines():
        t = RE_SPACE.sub(" ", line.strip())
        if not t:
            continue
        tl = t.lower()
        if tl.startswith(skip_prefixes):
            continue
        if t.startswith("http://") or t.startswith("https://"):
            continue
        kept.append(t)
        if len(" ".join(kept)) >= max_len:
            break

    return clean_description(" ".join(kept), max_len=max_len)


def is_probable_article_url(src: "ListingSource", url: str) -> bool:
    """
    Exclude listing landing pages while allowing true article URLs.
    """
    try:
        src_path = (urlparse(src.url).path or "").rstrip("/")
        path = (urlparse(url).path or "").rstrip("/")

        # Same path as listing source -> it's the landing page, not an article.
        if path == src_path:
            return False

        segs = [x for x in path.split("/") if x]

        # Reject common listing/tag hubs unless path is clearly article-like.
        if segs and segs[0] in {"tag", "tags", "topic", "topics", "hub", "section", "sections"}:
            if not re.search(r"/20\d{2}/\d{1,2}/\d{1,2}/", path) and "/article/" not in path:
                return False

        # Generic landing pages are usually shallow and evergreen.
        if len(segs) <= 2 and not re.search(r"/20\d{2}/\d{1,2}/\d{1,2}/", path):
            return False

        # Article-like if it has a date path, article/id marker, or long slug.
        if re.search(r"/20\d{2}/\d{1,2}/\d{1,2}/", path):
            return True
        if any(k in path for k in ("/article/", "-", ".php", ".html")):
            return True
        return len(segs) >= 3
    except Exception:
        return True


def looks_like_video_url(url: str) -> bool:
    path = (urlparse(url).path or "").lower()
    return "/video/" in path or path.startswith("/video")


def looks_like_video_story(title: str, summary: str, url: str) -> bool:
    if looks_like_video_url(url):
        return True
    blob = norm_text(f"{title} {summary}")
    video_hints = (
        "video", "watch", "highlights", "clip", "recap video", "postgame live"
    )
    return any(h in blob for h in video_hints)


# -----------------------------
# URL resolution: Google News RSS article links
# -----------------------------
def decode_google_news_token_url(gn_url: str) -> str:
    """
    Decode Google News RSS token URLs like /rss/articles/<token> where possible,
    without making a network request.
    """
    u = (gn_url or "").strip()
    if not u:
        return ""
    try:
        p = urlparse(u)
        host = (p.netloc or "").lower()
        if "news.google.com" not in host:
            return ""

        parts = [x for x in (p.path or "").split("/") if x]
        token = ""
        if "articles" in parts:
            i = parts.index("articles")
            if i + 1 < len(parts):
                token = parts[i + 1]
        elif "read" in parts:
            i = parts.index("read")
            if i + 1 < len(parts):
                token = parts[i + 1]

        if not token:
            return ""

        token = token.split("?")[0].split("#")[0].strip()
        pad = "=" * ((4 - len(token) % 4) % 4)
        raw = base64.urlsafe_b64decode(token + pad)
        text = raw.decode("utf-8", errors="ignore")

        m = re.search(r"https?://[^\s\"'<>\\]+", text)
        if not m:
            return ""
        cand = canonicalize_url(m.group(0).strip())
        d = domain_of(cand)
        if not cand or not d or is_google_host(cand) or is_blocked_domain(d) or is_reference_domain(d):
            return ""
        return cand
    except Exception:
        return ""


def resolve_google_news_article(gn_url: str) -> str:
    """
    Turn a Google News RSS article URL into the real publisher URL.
    We do NOT scrape random summary links; we explicitly resolve the GN article.
    """
    u = (gn_url or "").strip()
    if not u:
        return ""

    # If we somehow got a direct google redirect url, decode it.
    u_dec = decode_google_redirect_url(u)
    if u_dec != u and u_dec.startswith("http") and not is_google_host(u_dec):
        return canonicalize_url(u_dec)

    # Fast path: decode tokenized Google News URLs directly.
    tok = decode_google_news_token_url(u)
    if tok:
        return tok

    try:
        r = safe_get(u, timeout=20)

        # Many flows end on google.com/url?url=<target>
        final = (r.url or "").strip()
        final_dec = decode_google_redirect_url(final)
        if final_dec.startswith("http") and not is_google_host(final_dec):
            return canonicalize_url(final_dec)

        # If final is still GN HTML, look inside page for outbound URL hints.
        html = r.text or ""

        # 1) Look for explicit google redirect params in HTML
        m = re.search(r"https?://www\.google\.com/url\?[^\"'<> ]+", html, flags=re.I)
        if m:
            cand = decode_google_redirect_url(m.group(0))
            if cand.startswith("http") and not is_google_host(cand):
                return canonicalize_url(cand)

        # 2) Some GN pages contain "data-n-au" with the outbound URL
        m2 = re.search(r'data-n-au="([^"]+)"', html, flags=re.I)
        if m2:
            cand = unquote(m2.group(1)).strip()
            if cand.startswith("http") and not is_google_host(cand):
                return canonicalize_url(cand)

        # 3) Fallback: scan for any https URL that isn't google/tracker and looks like an article
        for uu in RE_URL.findall(html):
            cand = uu.strip()
            cand = decode_google_redirect_url(cand)
            d = domain_of(cand)
            if (
                cand.startswith("http")
                and d
                and not is_google_host(cand)
                and not is_blocked_domain(d)
                and not is_reference_domain(d)
                and len((urlparse(cand).path or "").strip("/")) > 3
            ):
                return canonicalize_url(cand)

    except Exception:
        pass

    return ""


def _candidate_urls_from_text(blob: str) -> List[str]:
    text = (blob or "").strip()
    if not text:
        return []
    out: List[str] = []
    out.extend(m.strip() for m in RE_URL.findall(text))
    out.extend(m.strip() for m in re.findall(r"href=[\"']([^\"']+)[\"']", text, flags=re.I))
    return out


def _pick_best_candidate_url(candidates: List[str], expected_domains: Set[str]) -> str:
    cleaned: List[str] = []
    for raw in candidates:
        u = decode_google_redirect_url((raw or "").strip())
        u = re.sub(r"[\x00-\x1f\x7f]+", "", u)
        u = canonicalize_url(u)
        d = domain_of(u)
        if not u.startswith("http") or not d:
            continue
        if is_google_host(u) or is_blocked_domain(d) or is_reference_domain(d):
            continue
        cleaned.append(u)

    if not cleaned:
        return ""

    if expected_domains:
        for u in cleaned:
            d = domain_of(u)
            if any(d == ed or d.endswith("." + ed) for ed in expected_domains):
                return u

    return cleaned[0]


def resolve_google_news_entry_url(entry: Any, source_label: str) -> str:
    expected_domains = expected_domains_for_source(source_label)
    candidates: List[str] = []

    link = (entry.get("link") or "").strip()
    if link:
        via_article = resolve_google_news_article(link)
        if via_article:
            candidates.append(via_article)
        candidates.append(link)

    entry_id = (entry.get("id") or "").strip()
    if entry_id:
        candidates.append(entry_id)

    links = entry.get("links") or []
    if isinstance(links, list):
        for ld in links:
            if isinstance(ld, dict):
                href = (ld.get("href") or "").strip()
                if href:
                    candidates.append(href)

    for field in ("summary", "description"):
        candidates.extend(_candidate_urls_from_text(entry.get(field) or ""))

    return _pick_best_candidate_url(candidates, expected_domains)


# -----------------------------
# RSS parsing
# -----------------------------
def parse_entry_datetime(entry: Any) -> Optional[datetime]:
    for key in ("published_parsed", "updated_parsed"):
        st = getattr(entry, key, None)
        if st:
            try:
                return datetime.fromtimestamp(time.mktime(st), tz=timezone.utc)
            except Exception:
                pass

    for key in ("published", "updated"):
        val = entry.get(key)
        if val:
            try:
                dt = dtparser.parse(val)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass
    return None


def extract_publication_from_title(raw_title: str) -> Tuple[str, str]:
    """
    Google News RSS items often look like: "Headline - Publication"
    Return (headline, publication)
    """
    t = (raw_title or "").strip()
    parts = [p.strip() for p in t.split(" - ") if p.strip()]
    if len(parts) >= 2 and len(parts[-1]) <= 60:
        return " - ".join(parts[:-1]).strip(), parts[-1].strip()
    return t, ""


def extract_author_from_entry(entry: Any) -> str:
    """
    feedparser typically exposes:
      entry.get("author")
      entry.get("authors") -> list of dicts with name
    """
    a = (entry.get("author") or "").strip()
    if a:
        return a
    authors = entry.get("authors") or []
    if isinstance(authors, list) and authors:
        a0 = authors[0]
        if isinstance(a0, dict):
            name = (a0.get("name") or "").strip()
            if name:
                return name
        if isinstance(a0, str):
            return a0.strip()
    return ""


def fetch_rss_items(
    feed_url: str,
    source_label: str,
    cutoff: datetime,
    full_names: Set[str],
    last_map: Dict[str, Set[str]],
) -> Tuple[List[Item], Dict[str, int]]:
    r = safe_get(feed_url, timeout=30)
    r.raise_for_status()

    fp = feedparser.parse(r.text)
    items: List[Item] = []

    total_entries = len(fp.entries)
    after_cutoff = 0
    after_url_domain = 0
    after_relevance = 0

    is_gn = looks_like_google_news_rss(feed_url)

    entries = fp.entries
    if is_gn and MAX_GOOGLE_ENTRIES_PER_FEED > 0:
        entries = entries[:MAX_GOOGLE_ENTRIES_PER_FEED]

    for e in entries:
        dt = parse_entry_datetime(e)
        if not dt or dt < cutoff:
            if DEBUG_REJECTIONS:
                print(f"[debug][reject][{source_label}] date_cutoff title={(e.get('title') or '')[:90]}")
            continue
        after_cutoff += 1

        raw_title = (e.get("title") or "").strip()
        if not raw_title:
            if DEBUG_REJECTIONS:
                print(f"[debug][reject][{source_label}] empty_title")
            continue

        headline, pub_from_title = extract_publication_from_title(raw_title)
        publication = pub_from_title or source_label

        author = extract_author_from_entry(e)

        # URL resolution rules:
        # - For Google News RSS: resolve the GN article page to the real publisher URL.
        # - For normal RSS: use entry.link and canonicalize.
        url = ""
        if is_gn:
            url = resolve_google_news_entry_url(e, source_label)
        else:
            url = canonicalize_url((e.get("link") or "").strip())

        if not url:
            if DEBUG_REJECTIONS:
                print(f"[debug][reject][{source_label}] no_url title={headline[:90]}")
            continue

        d = domain_of(url)
        if not d or is_google_host(url) or is_blocked_domain(d):
            if DEBUG_REJECTIONS:
                print(f"[debug][reject][{source_label}] blocked_domain domain={d or 'none'} title={headline[:90]} url={url[:120]}")
            continue

        expected_domains = expected_domains_for_source(source_label)
        if expected_domains and not any(d == ed or d.endswith("." + ed) for ed in expected_domains):
            if DEBUG_REJECTIONS:
                print(f"[debug][reject][{source_label}] domain_mismatch got={d} expected={sorted(expected_domains)} title={headline[:90]}")
            continue

        # GN targeted feeds can still surface home/section roots; reject those.
        if is_gn and source_label == "Google News: SFGiants.com / MLB Giants":
            if is_home_or_section_root_url(url) or not is_likely_story_url(url):
                if DEBUG_REJECTIONS:
                    print(f"[debug][reject][{source_label}] non_story_url url={url[:140]} title={headline[:90]}")
                continue

        if not is_strict_story_url_for_source(source_label, url, headline):
            if DEBUG_REJECTIONS:
                print(f"[debug][reject][{source_label}] strict_source_filter url={url[:140]} title={headline[:90]}")
            continue

        after_url_domain += 1

        summary = e.get("summary", "") or e.get("description", "") or ""

        allowed, reason = is_allowed_item(headline, summary, publication, d, full_names, last_map)
        if not allowed:
            if DEBUG_REJECTIONS:
                print(f"[debug][reject][{source_label}] {reason} domain={d} title={headline[:90]}")
            continue
        after_relevance += 1

        is_primary = any(d == pd or d.endswith("." + pd) for pd in PRIMARY_DOMAINS)

        items.append(
            Item(
                title=headline,
                url=url,
                publication=publication,
                published=dt,
                domain=d,
                is_primary=is_primary,
                author=author,
                raw_summary=summary,
            )
        )

    metrics = {
        "total_entries": total_entries,
        "after_cutoff": after_cutoff,
        "after_url_domain": after_url_domain,
        "after_relevance": after_relevance,
    }
    return items, metrics


# -----------------------------
# Listing page sources (non-RSS)
# -----------------------------
@dataclass
class ListingSource:
    name: str
    url: str
    domain: str
    # only keep links that match at least one of these path substrings/regexes
    allow_patterns: List[str]


def link_allowed_for_listing(src: ListingSource, url: str) -> bool:
    if not url:
        return False
    d = domain_of(url)
    if not d:
        return False
    if d != src.domain and not d.endswith("." + src.domain):
        return False
    # avoid trackers and weird scripts
    if is_blocked_domain(d):
        return False
    # must match allow patterns
    path = urlparse(url).path or ""
    for pat in src.allow_patterns:
        try:
            if pat.startswith("re:"):
                if re.search(pat[3:], path):
                    return True
            else:
                if pat in path:
                    return True
        except Exception:
            continue
    return False


@dataclass
class Candidate:
    url: str
    discovered_at: datetime
    discovered_from: str
    title: str = ""
    author: str = ""
    published: Optional[datetime] = None
    summary: str = ""


def parse_sitemap_xml(xml: str) -> Tuple[List[str], List[Tuple[str, Optional[datetime]]]]:
    sitemap_urls: List[str] = []
    story_urls: List[Tuple[str, Optional[datetime]]] = []
    if not xml:
        return sitemap_urls, story_urls

    for loc in re.findall(r"<loc>(.*?)</loc>", xml, flags=re.I | re.S):
        u = html_lib.unescape((loc or "").strip())
        if not u:
            continue
        if re.search(r"sitemap", u, flags=re.I):
            sitemap_urls.append(u)
        else:
            story_urls.append((u, None))

    for block in re.findall(r"<url>(.*?)</url>", xml, flags=re.I | re.S):
        lm = None
        mloc = re.search(r"<loc>(.*?)</loc>", block, flags=re.I | re.S)
        if not mloc:
            continue
        loc = html_lib.unescape(mloc.group(1).strip())
        mlm = re.search(r"<lastmod>(.*?)</lastmod>", block, flags=re.I | re.S)
        if mlm:
            try:
                dtx = dtparser.parse(html_lib.unescape(mlm.group(1).strip()))
                if dtx.tzinfo is None:
                    dtx = dtx.replace(tzinfo=timezone.utc)
                lm = dtx.astimezone(timezone.utc)
            except Exception:
                lm = None
        story_urls.append((loc, lm))
    return sitemap_urls, story_urls


def _discover_feeds_from_html(base_url: str, html: str) -> List[str]:
    out: List[str] = []
    for m in re.findall(r"<link[^>]+rel=[\"'][^\"']*alternate[^\"']*[\"'][^>]+>", html or "", flags=re.I):
        if re.search(r"application/(rss\+xml|atom\+xml)", m, flags=re.I):
            h = re.search(r"href=[\"']([^\"']+)[\"']", m, flags=re.I)
            if h:
                out.append(urljoin(base_url, h.group(1).strip()))
    return [canonicalize_url(x) for x in out if x]


def _common_feed_endpoints(base_url: str) -> List[str]:
    return [
        urljoin(base_url, p) for p in (
            "/feed", "/feed/", "/feed.xml", "/rss", "/rss.xml", "/atom.xml", "/index.xml",
        )
    ]


def _discover_sitemaps(domain_url: str, robots_txt: str) -> List[str]:
    found: List[str] = []
    for line in (robots_txt or "").splitlines():
        if line.lower().startswith("sitemap:"):
            u = line.split(":", 1)[1].strip()
            if u:
                found.append(canonicalize_url(u))
    found.append(urljoin(domain_url, "/sitemap.xml"))
    found.append(urljoin(domain_url, "/news-sitemap.xml"))
    ded: List[str] = []
    seen: Set[str] = set()
    for u in found:
        if u and u not in seen:
            seen.add(u)
            ded.append(u)
    return ded


def _cached_redirect(url: str, state: Dict[str, Any]) -> str:
    cache = state.setdefault("redirect_cache", {})
    now = utcnow()
    row = cache.get(url)
    if row:
        try:
            ts = dtparser.isoparse(row.get("ts", ""))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if (now - ts) < timedelta(hours=24):
                return row.get("final", url)
        except Exception:
            pass
    final = url
    try:
        r = safe_get(url, timeout=15)
        final = canonicalize_url(r.url or url)
    except Exception:
        final = url
    cache[url] = {"final": final, "ts": now.isoformat()}
    return final


def _build_item_from_candidate(
    src: ListingSource,
    cand: Candidate,
    cutoff: datetime,
    full_names: Set[str],
    last_map: Dict[str, Set[str]],
    meta_fetch_budget: Dict[str, int],
) -> Tuple[Optional[Item], str, bool]:
    url = canonicalize_url(cand.url)
    d = domain_of(url)
    if not d or is_google_host(url) or is_blocked_domain(d):
        return None, "blocked_domain", False
    if d != src.domain and not d.endswith("." + src.domain):
        return None, "domain_mismatch", False
    if not is_probable_article_url(src, url):
        return None, "not_probable_story", False

    meta: Dict[str, str] = {}
    text_fallback = ""
    meta_attempted = False
    if meta_fetch_budget["remaining"] > 0:
        meta_fetch_budget["remaining"] -= 1
        meta_attempted = True
        try:
            rr = safe_get(url, timeout=20)
            rr.raise_for_status()
            meta = parse_html_metadata(url, rr.text or "")
        except Exception:
            text_fallback = fetch_text_via_jina(url, timeout=20)

    canonical = canonicalize_url(meta.get("canonical") or url)
    d2 = domain_of(canonical)
    if d2 and (d2 == src.domain or d2.endswith("." + src.domain)) and not is_blocked_domain(d2):
        url = canonical
        d = d2

    title = (cand.title or meta.get("title") or "").strip()
    if not title and text_fallback:
        title = fallback_title_from_text(text_fallback, url)
    if not title:
        title = title_from_url(url)

    desc = (cand.summary or meta.get("description") or "").strip()
    if not desc and text_fallback:
        desc = fallback_description_from_text(text_fallback)

    author = (cand.author or meta.get("author") or "").strip()

    if looks_like_video_story(title, desc, url):
        return None, "video_story", meta_attempted

    published = cand.published
    if not published:
        pub_raw = (meta.get("published") or "").strip()
        if pub_raw:
            try:
                pdt = dtparser.parse(pub_raw)
                if pdt.tzinfo is None:
                    pdt = pdt.replace(tzinfo=timezone.utc)
                published = pdt.astimezone(timezone.utc)
            except Exception:
                published = None
    if not published:
        published = date_from_url(url)
    if not published:
        published = cand.discovered_at

    if published < cutoff:
        return None, "older_than_cutoff", meta_attempted

    if not is_strict_story_url_for_source(src.name, url, title):
        return None, "strict_source_filter", meta_attempted

    allowed, reason = is_allowed_item(title or url, desc, src.name, d, full_names, last_map)
    if not allowed:
        return None, reason, meta_attempted

    it = Item(
        title=title or url,
        url=url,
        publication=src.name,
        published=published,
        domain=d,
        is_primary=any(d == pd or d.endswith("." + pd) for pd in PRIMARY_DOMAINS),
        author=author,
        raw_summary=desc,
    )
    return it, "accepted", meta_attempted


def fetch_listing_items(
    src: ListingSource,
    cutoff: datetime,
    full_names: Set[str],
    last_map: Dict[str, Set[str]],
    meta_fetch_budget: Dict[str, int],
    state: Dict[str, Any],
) -> Tuple[List[Item], Dict[str, int]]:
    """Tiered ingest for non-RSS publishers.

    Why this is better: we don't rely on one fragile HTML <a> scrape path.
    We attempt native feed discovery, sitemaps, JSON/embed extraction, Google fallbacks,
    then HTML link scraping as last resort. Shared normalization/relevance stays centralized.
    """
    metrics: Dict[str, int] = {
        "tier_a_attempts": 0,
        "tier_b_attempts": 0,
        "tier_c_attempts": 0,
        "tier_d_attempts": 0,
        "tier_e_attempts": 0,
        "candidates_total": 0,
        "accepted": 0,
        "meta_attempts": 0,
        "meta_success": 0,
    }
    rejection_counts: Dict[str, int] = {}
    candidates: List[Candidate] = []
    seen: Set[str] = set()

    # Tier A: Native feeds (known + hidden discovery)
    metrics["tier_a_attempts"] += 1
    listing_html = ""
    feed_urls: List[str] = []
    try:
        lr = safe_get(src.url, timeout=20)
        lr.raise_for_status()
        listing_html = lr.text or ""
        feed_urls.extend(_discover_feeds_from_html(src.url, listing_html))
    except Exception:
        pass
    feed_urls.extend(_common_feed_endpoints(src.url))
    for fu in feed_urls[:10]:
        try:
            fr = safe_get(fu, timeout=15)
            if fr.status_code >= 400:
                continue
            fp = feedparser.parse(fr.text)
            if not fp.entries:
                continue
            for e in fp.entries[:MAX_LISTING_LINKS_PER_SOURCE]:
                dt = parse_entry_datetime(e) or utcnow()
                if dt < cutoff:
                    continue
                link = canonicalize_url((e.get("link") or "").strip())
                if not link or link in seen:
                    continue
                seen.add(link)
                candidates.append(Candidate(url=link, discovered_at=utcnow(), discovered_from=f"A:{fu}", title=(e.get("title") or "").strip(), summary=(e.get("summary") or "")))
        except Exception:
            continue

    # Tier B: Sitemaps
    metrics["tier_b_attempts"] += 1
    sitemap_cache = state.setdefault("sitemap_cache", {})
    cached = sitemap_cache.get(src.domain, {})
    story_urls: List[Tuple[str, Optional[datetime]]] = []
    use_cache = False
    if cached:
        try:
            ts = dtparser.isoparse(cached.get("fetched_at", ""))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if (utcnow() - ts) < timedelta(hours=6):
                use_cache = True
                story_urls = [(u, None) for u in cached.get("urls", [])]
        except Exception:
            pass

    if not use_cache:
        root = f"https://{src.domain}"
        robots = ""
        try:
            rr = safe_get(urljoin(root, "/robots.txt"), timeout=12)
            if rr.status_code < 400:
                robots = rr.text or ""
        except Exception:
            pass
        sm_queue = _discover_sitemaps(root, robots)
        visited_sm: Set[str] = set()
        while sm_queue and len(visited_sm) < 20:
            su = sm_queue.pop(0)
            if su in visited_sm:
                continue
            visited_sm.add(su)
            try:
                sr = safe_get(su, timeout=15)
                if sr.status_code >= 400:
                    continue
                child_maps, urls = parse_sitemap_xml(sr.text or "")
                # prioritize news sitemaps
                child_maps.sort(key=lambda x: 0 if "news" in x.lower() else 1)
                for c in child_maps:
                    if c not in visited_sm:
                        sm_queue.append(c)
                story_urls.extend(urls)
            except Exception:
                continue
        sitemap_cache[src.domain] = {
            "fetched_at": utcnow().isoformat(),
            "urls": [u for u, _ in story_urls[:2000]],
        }

    for u, lm in story_urls[:400]:
        u2 = canonicalize_url(u)
        if not u2 or u2 in seen:
            continue
        if lm and lm < cutoff:
            continue
        seen.add(u2)
        candidates.append(Candidate(url=u2, discovered_at=utcnow(), discovered_from="B:sitemap", published=lm))

    # Tier C: Embedded JSON / endpoints
    metrics["tier_c_attempts"] += 1
    if not listing_html:
        try:
            lr2 = safe_get(src.url, timeout=20)
            if lr2.status_code < 400:
                listing_html = lr2.text or ""
        except Exception:
            listing_html = ""
    if listing_html:
        for u in extract_links_from_listing(src.url, listing_html):
            u2 = canonicalize_url(u)
            if not u2 or u2 in seen:
                continue
            if domain_of(u2).endswith(src.domain):
                seen.add(u2)
                candidates.append(Candidate(url=u2, discovered_at=utcnow(), discovered_from="C:embedded"))
        # Next.js and embedded state url scan
        for u in _candidate_urls_from_text(listing_html):
            u2 = canonicalize_url(u)
            if not u2 or u2 in seen:
                continue
            if domain_of(u2).endswith(src.domain):
                seen.add(u2)
                candidates.append(Candidate(url=u2, discovered_at=utcnow(), discovered_from="C:state"))
        # WordPress REST endpoint
        try:
            wp = urljoin(f"https://{src.domain}", "/wp-json/wp/v2/posts?search=san%20francisco%20giants&per_page=20&_fields=link,date,title.rendered,excerpt.rendered")
            wr = safe_get(wp, timeout=15)
            if wr.status_code < 400:
                arr = wr.json()
                if isinstance(arr, list):
                    for row in arr:
                        if not isinstance(row, dict):
                            continue
                        link = canonicalize_url(str(row.get("link") or "").strip())
                        if not link or link in seen:
                            continue
                        pd = None
                        dv = str(row.get("date") or "").strip()
                        if dv:
                            try:
                                pd = dtparser.parse(dv)
                                if pd.tzinfo is None:
                                    pd = pd.replace(tzinfo=timezone.utc)
                                pd = pd.astimezone(timezone.utc)
                            except Exception:
                                pd = None
                        ttl = ""
                        tv = row.get("title")
                        if isinstance(tv, dict):
                            ttl = str(tv.get("rendered") or "")
                        summ = ""
                        ev = row.get("excerpt")
                        if isinstance(ev, dict):
                            summ = str(ev.get("rendered") or "")
                        seen.add(link)
                        candidates.append(Candidate(url=link, discovered_at=utcnow(), discovered_from="C:wp-json", title=ttl, summary=summ, published=pd))
        except Exception:
            pass

    # Tier D: Google feeds tied to this domain (search-based fallback)
    metrics["tier_d_attempts"] += 1
    gq = google_news_rss_url(f'(("San Francisco Giants" OR "SF Giants") AND (MLB OR baseball)) site:{src.domain}')
    try:
        gis, _ = fetch_rss_items(gq, f"Google News: {src.name}", cutoff, full_names, last_map)
        for it in gis[:MAX_LISTING_LINKS_PER_SOURCE]:
            if it.url in seen:
                continue
            seen.add(it.url)
            candidates.append(Candidate(url=it.url, discovered_at=utcnow(), discovered_from="D:google", title=it.title, author=it.author, published=it.published, summary=it.raw_summary))
    except Exception:
        pass

    # Tier E: HTML/text fallback
    metrics["tier_e_attempts"] += 1
    if listing_html:
        for u in extract_links_from_listing(src.url, listing_html):
            u2 = canonicalize_url(u)
            if not u2 or u2 in seen:
                continue
            seen.add(u2)
            candidates.append(Candidate(url=u2, discovered_at=utcnow(), discovered_from="E:html"))
    else:
        text = fetch_text_via_jina(src.url, timeout=20)
        for u in extract_urls_from_text_blob(text):
            u2 = canonicalize_url(u)
            if not u2 or u2 in seen:
                continue
            seen.add(u2)
            candidates.append(Candidate(url=u2, discovered_at=utcnow(), discovered_from="E:jina"))

    metrics["candidates_total"] = len(candidates)

    items: List[Item] = []
    source_seen = state.setdefault("source_seen_urls", {}).setdefault(src.name, {})

    for cand in candidates[:max(MAX_LISTING_LINKS_PER_SOURCE * 4, 60)]:
        if source_seen.get(cand.url):
            rejection_counts["already_seen_source"] = rejection_counts.get("already_seen_source", 0) + 1
            continue

        final_url = _cached_redirect(cand.url, state)
        cand.url = final_url
        it, reason, meta_attempted = _build_item_from_candidate(src, cand, cutoff, full_names, last_map, meta_fetch_budget)
        if meta_attempted:
            metrics["meta_attempts"] += 1
        if not it:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue

        metrics["accepted"] += 1
        if meta_attempted:
            metrics["meta_success"] += 1
        items.append(it)
        source_seen[it.url] = utcnow().isoformat()

    # prune per-source seen cache
    if len(source_seen) > 5000:
        keys = sorted(source_seen.items(), key=lambda kv: kv[1], reverse=True)
        state["source_seen_urls"][src.name] = dict(keys[:2500])

    metrics["rejections"] = sum(rejection_counts.values())
    for k, v in rejection_counts.items():
        metrics[f"reject_{k}"] = v
    return items, metrics


# -----------------------------
# Bluesky posting
# -----------------------------
def bsky_create_session() -> Dict[str, Any]:
    r = requests.post(
        f"{BSKY_PDS}/xrpc/com.atproto.server.createSession",
        json={"identifier": BSKY_IDENTIFIER, "password": BSKY_APP_PASSWORD},
        timeout=20,
        headers={"User-Agent": UA},
    )
    r.raise_for_status()
    return r.json()


def build_display_line(it: Item) -> str:
    label = (it.author or "").strip() or (it.publication or "").strip()
    t = (it.title or "").strip()

    # If title is accidentally a URL, derive a cleaner title from item URL.
    if looks_like_url(t):
        t = title_from_url(it.url)

    if not t:
        t = title_from_url(it.url)

    if is_paywalled_domain(it.domain):
        t = f"{t} ($)"

    if label:
        return f"{label}: {t}"
    return t


def clean_description(s: str, max_len: int = 240) -> str:
    txt = html_lib.unescape(s or "")
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = RE_SPACE.sub(" ", txt).strip()
    if len(txt) > max_len:
        txt = txt[: max_len - 1].rstrip() + "…"
    return txt


def build_post_text(it: Item) -> str:
    line = build_display_line(it)
    if len(line) <= 300:
        return line
    return line[:299].rstrip() + "…"


def build_external_embed(it: Item) -> Optional[Dict[str, Any]]:
    """
    Create an external embed card WITHOUT uploading a thumbnail blob.
    This avoids BlobTooLarge while still producing a "card" consistently.
    """
    if not ENABLE_EXTERNAL_EMBED:
        return None

    uri = (it.url or "").strip()
    if not uri:
        return None

    title = (it.title or "").strip()
    if not title:
        title = uri

    # Prefer actual article summary/description, cleaned.
    desc = clean_description(it.raw_summary or "")

    # If we have no description, a small fallback is better than empty.
    if not desc:
        # Use label + title lightly
        desc = (it.publication or "").strip()
        desc = desc[:240].strip()

    return {
        "$type": "app.bsky.embed.external",
        "external": {
            "uri": uri,
            "title": title[:300],
            "description": desc[:1000],
            # no thumb -> no blob -> avoids BlobTooLarge
        },
    }


def bsky_post(access_jwt: str, did: str, text: str, embed: Optional[Dict[str, Any]] = None) -> None:
    record: Dict[str, Any] = {
        "$type": "app.bsky.feed.post",
        "text": text,
        "createdAt": utcnow().isoformat().replace("+00:00", "Z"),
    }
    if embed:
        record["embed"] = embed

    r = requests.post(
        f"{BSKY_PDS}/xrpc/com.atproto.repo.createRecord",
        json={"repo": did, "collection": "app.bsky.feed.post", "record": record},
        timeout=20,
        headers={"Authorization": f"Bearer {access_jwt}", "User-Agent": UA},
    )
    if r.status_code >= 400:
        print(f"[error] createRecord {r.status_code}: {r.text[:2000]}")
    r.raise_for_status()


# -----------------------------
# Main
# -----------------------------
def effective_max_posts_per_run() -> Optional[int]:
    if MAX_POSTS_PER_RUN <= 0:
        return None
    return MAX_POSTS_PER_RUN


def title_hash(title: str) -> str:
    t = re.sub(r"[^a-z0-9 ]+", "", (title or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def main(dry_run: bool = DRY_RUN) -> None:
    state = load_state()
    prune_state(state)
    daily_other = get_daily_other_counter(state)

    cutoff = compute_cutoff(state)
    print(f"[info] cutoff={cutoff.isoformat()} (hours_back={HOURS_BACK})")

    roster_names = load_cached_names(state, "roster_cache", ROSTER_CACHE_HOURS)
    if not roster_names:
        try:
            roster_names = statsapi_40man_names()
            save_cached_names(state, "roster_cache", roster_names)
            print(f"[info] roster cached: {len(roster_names)} names")
        except Exception as e:
            print(f"[warn] roster fetch failed: {e}")
            roster_names = set()

    staff_names = load_cached_names(state, "staff_cache", STAFF_CACHE_HOURS)
    if not staff_names:
        try:
            staff_names = statsapi_coaches_names()
            save_cached_names(state, "staff_cache", staff_names)
            print(f"[info] staff cached: {len(staff_names)} names")
        except Exception as e:
            print(f"[warn] staff fetch failed: {e}")
            staff_names = set()

    all_names: Set[str] = set()
    all_names.update(roster_names)
    all_names.update(staff_names)
    all_names.update(KEY_PEOPLE)
    full_names, last_map = build_name_matchers(all_names)

    # -----------------------------
    # Sources
    # -----------------------------
    rss_feeds: List[Tuple[str, str]] = [
        ("SF Standard", "https://sfstandard.com/sports/feed"),
        ("SFGate", "https://www.sfgate.com/sports/feed/san-francisco-giants-rss-feed-428.php"),
    ]

    # Google News per-domain RSS (now with robust resolving)
    domain_queries = [
        ("SF Chronicle", '(("San Francisco Giants" OR "SF Giants" OR Giants) AND (MLB OR baseball)) site:sfchronicle.com'),
        ("Mercury News", '(("San Francisco Giants" OR "SF Giants" OR Giants) AND (MLB OR baseball)) site:mercurynews.com'),
        ("NBC Sports Bay Area", '(("San Francisco Giants" OR "SF Giants" OR Giants) AND (MLB OR baseball)) site:nbcsportsbayarea.com'),
        ("The Athletic", '(("San Francisco Giants" OR "SF Giants" OR Giants) AND (MLB OR baseball)) site:theathletic.com'),
        ("Associated Press", '(("San Francisco Giants" OR "SF Giants" OR Giants) AND (MLB OR baseball)) site:apnews.com'),
        ("SFGiants.com / MLB Giants", '(("San Francisco Giants" OR "SF Giants" OR Giants) AND (MLB OR baseball)) (site:mlb.com/giants OR site:sfgiants.com)'),
        ("FanGraphs", '(("San Francisco Giants" OR "SF Giants" OR Giants) AND (MLB OR baseball)) site:fangraphs.com'),
        ("Baseball America", '(("San Francisco Giants" OR "SF Giants" OR Giants) AND (MLB OR baseball)) site:baseballamerica.com'),
        ("KNBR", '(("San Francisco Giants" OR "SF Giants" OR Giants) AND (MLB OR baseball)) site:knbr.com'),
    ]
    for name, q in domain_queries:
        rss_feeds.append((f"Google News: {name}", google_news_rss_url(q)))

    rss_feeds.append(("Google News: Broad", google_news_rss_url('(("San Francisco Giants" OR "SF Giants") AND (MLB OR baseball))')))
    for i, alert_feed in enumerate(GOOGLE_ALERTS_RSS):
        rss_feeds.append((f"Google Alerts #{i+1}", alert_feed))

    # Non-RSS listing pages (the “radical” part).
    # Patterns are intentionally conservative to avoid pulling unrelated site links.
    listing_sources: List[ListingSource] = [
        ListingSource(
            name="SF Chronicle",
            url="https://www.sfchronicle.com/sports/giants/",
            domain="sfchronicle.com",
            allow_patterns=["/sports/giants/", "re:/\d{4}/\d{2}/\d{2}/"],
        ),
        ListingSource(
            name="Mercury News",
            url="https://www.mercurynews.com/tag/san-francisco-giants/",
            domain="mercurynews.com",
            allow_patterns=["/tag/san-francisco-giants/", "re:/\d{4}/\d{2}/\d{2}/"],
        ),
        ListingSource(
            name="AP News",
            url="https://apnews.com/hub/san-francisco-giants",
            domain="apnews.com",
            allow_patterns=["/article/", "/hub/san-francisco-giants"],
        ),
        ListingSource(
            name="NBC Sports Bay Area",
            url="https://www.nbcsportsbayarea.com/mlb/san-francisco-giants/",
            domain="nbcsportsbayarea.com",
            allow_patterns=["/mlb/san-francisco-giants/", "re:/\d{6,}/$"],
        ),
        ListingSource(
            name="KNBR Giants",
            url="https://www.knbr.com/category/san-francisco-giants/",
            domain="knbr.com",
            allow_patterns=["/san-francisco-giants/", "re:/\d{4}/\d{2}/\d{2}/"],
        ),
        ListingSource(
            name="FanGraphs Giants",
            url="https://blogs.fangraphs.com/category/giants/",
            domain="fangraphs.com",
            allow_patterns=["/category/giants/", "re:/\d{4}/\d{2}/"],
        ),
        ListingSource(
            name="Baseball America Giants",
            url="https://www.baseballamerica.com/teams/mlb/san-francisco-giants/",
            domain="baseballamerica.com",
            allow_patterns=["/teams/mlb/san-francisco-giants/", "re:/story/", "re:/news/"],
        ),
        ListingSource(
            name="The Athletic Giants",
            url="https://www.nytimes.com/athletic/mlb/team/giants/",
            domain="nytimes.com",
            allow_patterns=["/athletic/", "re:/athletic/\d+/"],
        ),
        ListingSource(
            name="MLB Giants News",
            url="https://www.mlb.com/giants/news",
            domain="mlb.com",
            allow_patterns=["/giants/news/", "re:/news/.+"],
        ),
        ListingSource(
            name="SFGiants.com News",
            url="https://www.sfgiants.com/news",
            domain="sfgiants.com",
            allow_patterns=["/news/", "re:/news/.+"],
        ),
    ]


    # -----------------------------
    # Collect items
    # -----------------------------
    all_items: List[Item] = []
    successful_sources = 0

    # RSS path
    for source_label, feed_url in rss_feeds:
        try:
            items, metrics = fetch_rss_items(feed_url, source_label, cutoff, full_names, last_map)
            successful_sources += 1
            print(
                "[info] rss metrics"
                f" source={source_label}"
                f" total={metrics['total_entries']}"
                f" after_cutoff={metrics['after_cutoff']}"
                f" after_url_domain={metrics['after_url_domain']}"
                f" after_relevance={metrics['after_relevance']}"
            )
            all_items.extend(items)
        except Exception as ex:
            print(f"[warn] rss feed failed: {source_label}: {ex}")

    # Listing path (non-RSS). Separate budget from embed enrichment so this path never turns off by accident.
    if ENABLE_LISTING_FETCH:
        meta_budget = {"remaining": MAX_LISTING_META_FETCHES_PER_RUN}
        for src in listing_sources:
            if meta_budget["remaining"] <= 0:
                print(f"[info] listing metrics source={src.name} skipped=budget_exhausted")
                continue
            try:
                per_source_budget = {"remaining": min(MAX_LISTING_META_FETCHES_PER_SOURCE, meta_budget["remaining"])}
                items, metrics = fetch_listing_items(src, cutoff, full_names, last_map, per_source_budget, state)
                meta_budget["remaining"] -= metrics.get("meta_attempts", 0)
                successful_sources += 1
                print(
                    "[info] listing metrics"
                    f" source={src.name}"
                    f" candidates_total={metrics.get('candidates_total', 0)}"
                    f" accepted={metrics.get('accepted', 0)}"
                    f" tierA={metrics.get('tier_a_attempts', 0)}"
                    f" tierB={metrics.get('tier_b_attempts', 0)}"
                    f" tierC={metrics.get('tier_c_attempts', 0)}"
                    f" tierD={metrics.get('tier_d_attempts', 0)}"
                    f" tierE={metrics.get('tier_e_attempts', 0)}"
                    f" meta_attempts={metrics.get('meta_attempts', 0)}"
                    f" meta_success={metrics.get('meta_success', 0)}"
                    f" rejections={metrics.get('rejections', 0)}"
                    f" budget_left={meta_budget['remaining']}"
                )
                all_items.extend(items)
            except Exception as ex:
                print(f"[warn] listing source failed: {src.name}: {ex}")
    else:
        print("[info] listing fetch disabled via ENABLE_LISTING_FETCH")

    posted = state.get("posted", {})
    seen_urls: Set[str] = set()

    if not all_items:
        print("No eligible items found.")
        if successful_sources > 0:
            state["last_run_success_at"] = utcnow().isoformat()
        save_state(state)
        return

    # Dedupe by canonicalized URL early
    deduped: List[Item] = []
    for it in all_items:
        it.url = canonicalize_url(it.url)
        if it.url in seen_urls:
            continue
        seen_urls.add(it.url)
        deduped.append(it)
    all_items = deduped

    # Echo counts
    echo_map: Dict[str, int] = {}
    for it in all_items:
        h = title_hash(it.title)
        echo_map[h] = echo_map.get(h, 0) + 1

    # Score
    for it in all_items:
        it.score = importance_score(
            it.title,
            it.raw_summary,
            it.publication,
            full_names,
            KEY_PEOPLE,
            it.published,
            echo_map.get(title_hash(it.title), 1),
        )

    # Newest first
    all_items.sort(key=lambda x: x.published, reverse=True)

    primary_candidates: List[Item] = []
    other_candidates: List[Item] = []
    for it in all_items:
        if it.url in posted:
            continue
        it.is_primary = any(it.domain == d or it.domain.endswith("." + d) for d in PRIMARY_DOMAINS)
        if it.is_primary:
            primary_candidates.append(it)
        else:
            other_candidates.append(it)

    other_candidates.sort(key=lambda x: (x.score, x.published), reverse=True)

    max_posts = effective_max_posts_per_run()
    to_post: List[Item] = []

    for it in primary_candidates:
        if max_posts is not None and len(to_post) >= max_posts:
            break
        to_post.append(it)

    other_budget = max(0, OTHER_DAILY_CAP - int(daily_other.get("count", 0)))
    for it in other_candidates:
        if max_posts is not None and len(to_post) >= max_posts:
            break
        if other_budget <= 0:
            break
        to_post.append(it)
        other_budget -= 1

    if ENFORCE_PER_SOURCE_CAP and PER_SOURCE_CAP > 0:
        filtered: List[Item] = []
        per_pub_count: Dict[str, int] = {}

        def pub_key(pub: str) -> str:
            return norm_text(pub)

        for it in to_post:
            pk = pub_key(it.publication)
            per_pub_count.setdefault(pk, 0)
            if per_pub_count[pk] >= PER_SOURCE_CAP:
                continue
            filtered.append(it)
            per_pub_count[pk] += 1
        to_post = filtered

    if not to_post:
        print("No new items to post.")
        if successful_sources > 0:
            state["last_run_success_at"] = utcnow().isoformat()
        save_state(state)
        return

    if dry_run:
        print("[dry-run] final items to post:")
        for it in to_post:
            print(f"[dry-run][item] source={it.publication} title={it.title} url={it.url}")
        if successful_sources > 0:
            state["last_run_success_at"] = utcnow().isoformat()
        save_state(state)
        return

    sess = bsky_create_session()
    access_jwt = sess["accessJwt"]
    did = sess["did"]

    posted_any = 0
    for it in to_post:
        text = build_post_text(it)
        embed = build_external_embed(it) if ENABLE_EXTERNAL_EMBED else None
        try:
            print(f"[post] ({it.score}) {it.publication}: {it.title} -> {it.url}")
            bsky_post(access_jwt, did, text, embed=embed)
            posted[it.url] = utcnow().isoformat()
            posted_any += 1
            if not it.is_primary:
                daily_other["count"] = int(daily_other.get("count", 0)) + 1
        except Exception as e:
            print(f"[warn] post failed: {e}")

    state["posted"] = posted
    state["daily_other"] = daily_other
    if successful_sources > 0:
        state["last_run_success_at"] = utcnow().isoformat()
    save_state(state)

    if posted_any == 0:
        raise SystemExit("All posts failed.")
    print(f"Posted {posted_any} items.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SF Giants Bluesky news bot")
    ap.add_argument("--dry-run", action="store_true", help="Run ingestion and ranking without posting")
    args = ap.parse_args()
    main(dry_run=args.dry_run or DRY_RUN)
