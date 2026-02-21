import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse

import feedparser
import requests
from dateutil import parser as dtparser


# -----------------------------
# Config
# -----------------------------
TEAM_ID = 137  # SF Giants (MLB Stats API)

HOURS_BACK = int(os.getenv("HOURS_BACK", "8"))
# 0 (or missing/invalid) => unlimited
MAX_POSTS_PER_RUN = int(os.getenv("MAX_POSTS_PER_RUN", "0") or "0")
STATE_FILE = os.getenv("STATE_FILE", "state.json")

# Per-run diversity is off by default. Enable explicitly if you want it.
ENFORCE_PER_SOURCE_CAP = os.getenv("ENFORCE_PER_SOURCE_CAP", "false").lower() in {"1", "true", "yes"}
PER_SOURCE_CAP = int(os.getenv("PER_SOURCE_CAP", "0") or "0")

# “Best of the rest” (outside your chosen domains): cap posts per day.
# This applies only to non-primary domains.
OTHER_DAILY_CAP = int(os.getenv("OTHER_DAILY_CAP", "2"))

# Cache lifetimes
ROSTER_CACHE_HOURS = int(os.getenv("ROSTER_CACHE_HOURS", "24"))
STAFF_CACHE_HOURS = int(os.getenv("STAFF_CACHE_HOURS", "24"))
KEEP_POSTED_DAYS = int(os.getenv("KEEP_POSTED_DAYS", "21"))

DEBUG_REJECTIONS = os.getenv("DEBUG_REJECTIONS", "0") == "1"

# Paywall tagging (based on your rule + practical access)
PAYWALL_DOMAINS = {
    "theathletic.com",
    "mercurynews.com",
    "baseballamerica.com",
    "sfchronicle.com",
}

BSKY_IDENTIFIER = os.environ["BSKY_IDENTIFIER"]
BSKY_APP_PASSWORD = os.environ["BSKY_APP_PASSWORD"]
BSKY_PDS = os.getenv("BSKY_PDS", "https://bsky.social")

UA = "GiantsNewsBot/2.4 (+github-actions)"


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

AGGREGATOR_BLOCKLIST = {
    "news.google.com",
    "feedspot.com",
    "feedly.com",
    "newsbreak.com",
    "ground.news",
}


# Google/CDN/asset hosts that often appear on Google News pages (not the real article)
ASSET_DOMAIN_BLOCKLIST = {
    "lh3.googleusercontent.com",
    "googleusercontent.com",
    "gstatic.com",
    "ggpht.com",
    "googleapis.com",
    "ytimg.com",
}

# Tracker/ad/script hosts that are never the article
TRACKER_DOMAIN_BLOCKLIST = {
    "google-analytics.com",
    "www.google-analytics.com",
    "googletagmanager.com",
    "www.googletagmanager.com",
    "doubleclick.net",
    "www.doubleclick.net",
    "googlesyndication.com",
    "www.googlesyndication.com",
    "adsystem.com",
    "adservice.google.com",
}

# File extensions that are almost never an article page
BAD_EXTENSIONS = {
    ".js",
    ".css",
    ".json",
    ".xml",
    ".rss",
    ".ico",
    ".svg",
    ".map",
}

ASSET_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".bmp",
    ".ico",
    ".mp4",
    ".mov",
    ".m4v",
    ".webm",
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
    "mlb",
    "major league",
    "baseball",
    "spring training",
    "cactus league",
    "grapefruit league",
    "opening day",
    "postseason",
    "playoffs",
    "world series",
    "nl west",
    "national league",
    "trade",
    "traded",
    "acquired",
    "deal",
    "deadline",
    "dfa",
    "designated for assignment",
    "waivers",
    "claimed",
    "optioned",
    "option",
    "call-up",
    "called up",
    "sent down",
    "roster",
    "40-man",
    "40 man",
    "injured list",
    "il",
    "rehab assignment",
    "pitcher",
    "starter",
    "rotation",
    "bullpen",
    "reliever",
    "closer",
    "catcher",
    "shortstop",
    "second base",
    "third base",
    "outfield",
    "first base",
    "dh",
    "inning",
    "innings",
    "era",
    "fip",
    "whip",
    "strikeout",
    "strikeouts",
    "walks",
    "ks",
    "home run",
    "homer",
    "batting",
    "slugging",
    "ops",
    "wrc+",
    "war",
    "xwoba",
    "prospect",
    "prospects",
    "farm system",
    "scouting",
    "draft",
    "international signing",
    "player development",
    "minor league",
    "triple-a",
    "double-a",
    "manager",
    "bench coach",
    "pitching coach",
    "hitting coach",
    "coach",
    "general manager",
    "gm",
    "front office",
    "president of baseball operations",
]

NEGATIVE_PHRASES = [
    "dear abby",
    "new york giants",
    "ny giants",
    "nfl",
    "super bowl",
    "touchdown",
    "quarterback",
    "wide receiver",
    "linebacker",
]

BASEBALL_SOURCE_HINTS = [
    "fangraphs",
    "baseball america",
    "mlb",
    "sfgiants",
    "baseball prospectus",
]

IMPORTANT_KEYWORDS = {
    "trade",
    "traded",
    "acquire",
    "acquired",
    "deal",
    "waiver",
    "waivers",
    "claimed",
    "dfa",
    "designated for assignment",
    "optioned",
    "call-up",
    "called up",
    "sign",
    "signed",
    "signing",
    "extension",
    "injury",
    "injured",
    "il",
    "injured list",
    "surgery",
    "rehab",
    "prospect",
    "prospects",
    "promotion",
    "promoted",
    "rotation",
    "bullpen",
    "starter",
    "closer",
}


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


RE_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
RE_SPACE = re.compile(r"\s+")
RE_META = re.compile(
    r'<meta[^>]+property=["\']([^"\']+)["\'][^>]+content=["\']([^"\']+)["\']',
    flags=re.I,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def norm_text(s: str) -> str:
    return RE_SPACE.sub(" ", (s or "").strip().lower())


def domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def safe_get(url: str, timeout: int = 25) -> requests.Response:
    return requests.get(url, timeout=timeout, headers={"User-Agent": UA}, allow_redirects=True)


def title_hash(title: str) -> str:
    t = re.sub(r"[^a-z0-9 ]+", "", (title or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def extract_publication_from_title(raw_title: str) -> Tuple[str, str]:
    t = (raw_title or "").strip()
    parts = [p.strip() for p in t.split(" - ") if p.strip()]
    if len(parts) >= 2 and len(parts[-1]) <= 60:
        return " - ".join(parts[:-1]).strip(), parts[-1].strip()
    return t, ""


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


def is_primary_domain(domain: str) -> bool:
    d = (domain or "").lower()
    return any(d == pd or d.endswith("." + pd) for pd in PRIMARY_DOMAINS)


def google_news_rss_url(query: str) -> str:
    q = quote(query)
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def is_google_host(url: str) -> bool:
    host = domain_of(url)
    return ("news.google.com" in host) or ("google.com" in host)


def decode_google_news_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u

    if not is_google_host(u):
        return u

    try:
        parsed = urlparse(u)
        q = parse_qs(parsed.query)
        for key in ("url", "u", "q", "redirect", "continue"):
            for val in (q.get(key) or []):
                cand = unquote(val).strip()
                if cand.startswith("http") and not is_google_host(cand):
                    return cand
    except Exception:
        pass

    # Sometimes the path itself contains an encoded URL.
    try:
        decoded_path = unquote(urlparse(u).path)
        m = re.search(r"(https?://[^\s]+)", decoded_path)
        if m and not is_google_host(m.group(1)):
            return m.group(1)
    except Exception:
        pass

    return u


def is_asset_like_url(u: str) -> bool:
    if not u:
        return True

    du = domain_of(u)
    if not du:
        return True

    # Block common CDN/asset and tracker hosts
    if (
        du in ASSET_DOMAIN_BLOCKLIST
        or any(du.endswith("." + x) for x in ASSET_DOMAIN_BLOCKLIST)
        or du in TRACKER_DOMAIN_BLOCKLIST
        or any(du.endswith("." + x) for x in TRACKER_DOMAIN_BLOCKLIST)
    ):
        return True

    path = urlparse(u).path.lower()

    # Reject obvious non-article file types
    if any(path.endswith(ext) for ext in BAD_EXTENSIONS):
        return True

    # Reject obvious media assets
    if any(path.endswith(ext) for ext in ASSET_EXTENSIONS):
        return True

    return False


def extract_href_candidates(html: str) -> List[str]:
    """
    Prefer href/meta-like candidates, not "any URL anywhere".
    This avoids grabbing analytics.js, pixels, etc.
    """
    if not html:
        return []

    cands: List[str] = []

    # Anchor/link hrefs
    for h in re.findall(r'href=["\'](https?://[^"\']+)["\']', html, flags=re.I):
        cands.append(h)

    # Meta refresh redirects
    for content in re.findall(
        r'<meta[^>]+http-equiv=["\']refresh["\'][^>]+content=["\']([^"\']+)["\']', html, flags=re.I
    ):
        m = re.search(r"url=(https?://\S+)", content, flags=re.I)
        if m:
            cands.append(m.group(1))

    # Meta og:url / twitter:url (may still be Google host; we score-filter later)
    for prop, val in RE_META.findall(html):
        p = (prop or "").lower()
        if p in {"og:url", "twitter:url"} and val:
            cands.append(val)

    # De-dupe preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for u in cands:
        u = (u or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def score_candidate_url(u: str) -> int:
    """
    Score URLs so we pick the publisher article, not scripts/trackers.
    """
    if not u or not u.startswith("http"):
        return -10_000

    u2 = decode_google_news_url(u)
    if is_google_host(u2):
        return -9_000

    if is_asset_like_url(u2):
        return -8_000

    d = domain_of(u2)
    if not d:
        return -7_000

    score = 0

    # Strong preference: your publisher domains
    if is_primary_domain(d):
        score += 5_000

    path = (urlparse(u2).path or "").lower()
    if any(x in path for x in ["/article", "/sports", "/mlb", "/giants", "/story", "/news"]):
        score += 300

    # Prefer longer paths
    score += min(200, len(path))

    lowered = u2.lower()
    if any(k in lowered for k in ["analytics", "tagmanager", "doubleclick", "pixel"]):
        score -= 5_000

    return score


def extract_best_external_url(html: str) -> str:
    """
    Find the best external URL in a Google News HTML page.
    Only consider href/meta-like candidates, then score them.
    """
    candidates = extract_href_candidates(html)
    if not candidates:
        return ""

    best_url = ""
    best_score = -10_000_000

    for c in candidates:
        c = decode_google_news_url(c)
        if not c or not c.startswith("http"):
            continue
        s = score_candidate_url(c)
        if s > best_score:
            best_score = s
            best_url = c

    return best_url


def pick_best_url(entry: Any) -> str:
    # Prefer links list first
    for l in entry.get("links", []) or []:
        href = l.get("href")
        if not href:
            continue
        decoded = decode_google_news_url(href)
        if decoded and not is_google_host(decoded) and not is_asset_like_url(decoded):
            return decoded

    # Next: description/summary hrefs
    summary = entry.get("summary", "") or entry.get("description", "") or ""
    hrefs = re.findall(r'href=["\'](https?://[^"\']+)["\']', summary, flags=re.I)
    for h in hrefs:
        d = decode_google_news_url(h)
        if d and not is_google_host(d) and not is_asset_like_url(d):
            return d

    fallback = decode_google_news_url(entry.get("link", "") or "")
    return fallback


def resolve_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u

    u = decode_google_news_url(u)
    if u and not is_google_host(u) and not is_asset_like_url(u):
        return u

    # Final fallback: fetch once and find best external candidate
    try:
        r = safe_get(u, timeout=20)
        if r.url:
            resolved = decode_google_news_url(r.url)
            if resolved and not is_google_host(resolved) and not is_asset_like_url(resolved):
                return resolved

        best = extract_best_external_url(r.text or "")
        if best:
            return best
    except Exception:
        pass

    return u


def is_paywalled_domain(domain: str) -> bool:
    d = (domain or "").lower()
    return any(d == pw or d.endswith("." + pw) for pw in PAYWALL_DOMAINS)


def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {
            "posted": {},
            "roster_cache": {},
            "staff_cache": {},
            "daily_other": {},
            "last_run_success_at": None,
        }
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)
    state.setdefault("posted", {})
    state.setdefault("roster_cache", {})
    state.setdefault("staff_cache", {})
    state.setdefault("daily_other", {})
    state.setdefault("last_run_success_at", None)
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
# MLB Stats API: roster + (best effort) coaches
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
def tokenize_words(text: str) -> Set[str]:
    return set(w.lower() for w in RE_WORD.findall(text or ""))


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

    # Recall boost: primary domains with "Giants" mention can pass even when
    # summary is thin and misses explicit MLB/baseball keywords.
    if is_primary_domain(domain) and contains_phrase(t, "giants"):
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
# Author extraction
# -----------------------------
def extract_author(entry: Any, summary_html: str) -> str:
    """
    Best-effort author extraction from RSS/Atom entries.
    Different feeds use different fields.
    """
    # Feedparser normalizes some
    author = (entry.get("author") or "").strip()
    if author:
        return author

    # Some have "authors" list
    authors = entry.get("authors") or []
    if isinstance(authors, list) and authors:
        # Each element might be dict with 'name'
        for a in authors:
            if isinstance(a, dict):
                nm = (a.get("name") or "").strip()
                if nm:
                    return nm

    # Sometimes embedded in summary as "By X"
    txt = re.sub(r"<[^>]+>", " ", summary_html or "")
    m = re.search(r"\bBy\s+([A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){0,3})\b", txt)
    if m:
        cand = (m.group(1) or "").strip()
        # Avoid "By Giants" etc
        if cand and cand.lower() not in {"giants", "sf", "mlb"}:
            return cand

    return ""


# -----------------------------
# Feeds
# -----------------------------
def fetch_feed_items(
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

    for e in fp.entries:
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

        raw_url = pick_best_url(e)
        url = resolve_url(raw_url)
        if not url:
            if DEBUG_REJECTIONS:
                print(f"[debug][reject][{source_label}] no_url title={headline[:90]}")
            continue

        d = domain_of(url)
        if not d or d in AGGREGATOR_BLOCKLIST or is_google_host(url) or is_asset_like_url(url):
            if DEBUG_REJECTIONS:
                print(f"[debug][reject][{source_label}] blocked_domain domain={d or 'none'} title={headline[:90]}")
            continue
        after_url_domain += 1

        summary = e.get("summary", "") or e.get("description", "") or ""
        author = extract_author(e, summary)

        allowed, reason = is_allowed_item(headline, summary, publication, d, full_names, last_map)
        if not allowed:
            if DEBUG_REJECTIONS:
                print(f"[debug][reject][{source_label}] {reason} domain={d} title={headline[:90]}")
            continue
        after_relevance += 1

        is_primary = is_primary_domain(d)
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
# Bluesky
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


def make_link_facet(text: str, url: str) -> List[Dict[str, Any]]:
    u = (url or "").strip()
    if not u:
        return []

    idx = text.rfind(u)
    if idx < 0:
        return []

    byte_start = len(text[:idx].encode("utf-8"))
    byte_end = byte_start + len(u.encode("utf-8"))
    return [
        {
            "index": {"byteStart": byte_start, "byteEnd": byte_end},
            "features": [{"$type": "app.bsky.richtext.facet#link", "uri": u}],
        }
    ]


def bsky_upload_blob(access_jwt: str, content_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    """
    Upload an image blob for external embeds.
    """
    r = requests.post(
        f"{BSKY_PDS}/xrpc/com.atproto.repo.uploadBlob",
        data=content_bytes,
        timeout=30,
        headers={
            "Authorization": f"Bearer {access_jwt}",
            "User-Agent": UA,
            "Content-Type": mime_type,
        },
    )
    r.raise_for_status()
    return r.json().get("blob") or {}


def fetch_external_card(url: str) -> Dict[str, Any]:
    """
    Fetch basic OpenGraph-like metadata from the target URL to build an external embed.
    If this fails, we still post without an embed.
    """
    try:
        r = safe_get(url, timeout=20)
        html = r.text or ""
    except Exception:
        return {}

    meta: Dict[str, str] = {}
    for prop, val in RE_META.findall(html):
        p = (prop or "").strip().lower()
        if not p or not val:
            continue
        if p in {"og:title", "og:description", "og:image", "og:site_name", "twitter:title", "twitter:description", "twitter:image"}:
            # Prefer og:* over twitter:* if both exist
            if p.startswith("twitter:"):
                og_equiv = "og:" + p.split(":", 1)[1]
                if og_equiv in meta:
                    continue
            meta[p] = val.strip()

    title = meta.get("og:title") or meta.get("twitter:title") or ""
    desc = meta.get("og:description") or meta.get("twitter:description") or ""
    img = meta.get("og:image") or meta.get("twitter:image") or ""

    # Keep within typical limits; Bluesky also has size limits.
    title = (title or "").strip()[:300]
    desc = (desc or "").strip()[:1000]
    img = (img or "").strip()

    out: Dict[str, Any] = {"uri": url}
    if title:
        out["title"] = title
    if desc:
        out["description"] = desc
    if img and img.startswith("http") and not is_asset_like_url(img):
        out["thumb_url"] = img
    return out


def build_external_embed(access_jwt: str, card: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert fetched metadata into app.bsky.embed.external.
    Downloads thumb (if present) and uploads as a blob.
    """
    if not card or not card.get("uri"):
        return None

    external: Dict[str, Any] = {
        "uri": card["uri"],
        "title": card.get("title") or card["uri"],
        "description": card.get("description") or "",
    }

    thumb_url = card.get("thumb_url")
    if thumb_url:
        try:
            img_r = safe_get(thumb_url, timeout=20)
            img_r.raise_for_status()
            mime = img_r.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
            blob = bsky_upload_blob(access_jwt, img_r.content, mime_type=mime)
            if blob:
                external["thumb"] = blob
        except Exception:
            pass

    return {"$type": "app.bsky.embed.external", "external": external}


def bsky_post(
    access_jwt: str,
    did: str,
    text: str,
    facets: Optional[List[Dict[str, Any]]] = None,
    embed: Optional[Dict[str, Any]] = None,
) -> None:
    record: Dict[str, Any] = {
        "$type": "app.bsky.feed.post",
        "text": text,
        "createdAt": utcnow().isoformat().replace("+00:00", "Z"),
    }
    if facets:
        record["facets"] = facets
    if embed:
        record["embed"] = embed

    r = requests.post(
        f"{BSKY_PDS}/xrpc/com.atproto.repo.createRecord",
        json={"repo": did, "collection": "app.bsky.feed.post", "record": record},
        timeout=30,
        headers={"Authorization": f"Bearer {access_jwt}", "User-Agent": UA},
    )
    if r.status_code >= 400:
        print(f"[error] createRecord {r.status_code}: {r.text[:2000]}")
    r.raise_for_status()


def display_prefix(it: Item) -> str:
    """
    If author is available: "Author: Title"
    else: "Publication: Title"
    """
    a = (it.author or "").strip()
    if a:
        return f"{a}: {it.title}".strip()
    return f"{(it.publication or '').strip()}: {it.title}".strip()


def format_post_text(it: Item) -> str:
    prefix_title = display_prefix(it)
    u = (it.url or "").strip()

    if is_paywalled_domain(it.domain):
        prefix_title = f"{prefix_title} ($)"

    text = f"{prefix_title}\n\n{u}".strip()
    if len(text) <= 300:
        return text

    room_for_title = max(20, 300 - (len(u) + 2))
    if len(prefix_title) > room_for_title:
        prefix_title = prefix_title[: room_for_title - 1].rstrip() + "…"
    return f"{prefix_title}\n\n{u}"


def effective_max_posts_per_run() -> Optional[int]:
    if MAX_POSTS_PER_RUN <= 0:
        return None
    return MAX_POSTS_PER_RUN


# -----------------------------
# Main
# -----------------------------
def main() -> None:
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

    feeds: List[Tuple[str, str]] = [
        ("SF Standard", "https://sfstandard.com/sports/feed"),
        ("SFGate", "https://www.sfgate.com/sports/feed/san-francisco-giants-rss-feed-428.php"),
    ]

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
        feeds.append((f"Google News: {name}", google_news_rss_url(q)))

    feeds.append(
        (
            "Google News: Broad",
            google_news_rss_url('(("San Francisco Giants" OR "SF Giants") AND (MLB OR baseball))'),
        )
    )

    all_items: List[Item] = []
    successful_feed_fetches = 0
    for source_label, feed_url in feeds:
        try:
            items, metrics = fetch_feed_items(feed_url, source_label, cutoff, full_names, last_map)
            successful_feed_fetches += 1
            print(
                "[info] feed metrics"
                f" source={source_label}"
                f" total={metrics['total_entries']}"
                f" after_cutoff={metrics['after_cutoff']}"
                f" after_url_domain={metrics['after_url_domain']}"
                f" after_relevance={metrics['after_relevance']}"
            )
            all_items.extend(items)
        except Exception as ex:
            print(f"[warn] feed failed: {source_label}: {ex}")

    posted = state.get("posted", {})
    seen_urls: Set[str] = set()

    if not all_items:
        print("No eligible items found.")
        if successful_feed_fetches > 0:
            state["last_run_success_at"] = utcnow().isoformat()
        save_state(state)
        return

    echo_map: Dict[str, int] = {}
    for it in all_items:
        h = title_hash(it.title)
        echo_map[h] = echo_map.get(h, 0) + 1

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
        if it.url in posted or it.url in seen_urls:
            continue
        seen_urls.add(it.url)

        it.is_primary = is_primary_domain(it.domain)
        if it.is_primary:
            primary_candidates.append(it)
        else:
            other_candidates.append(it)

    other_candidates.sort(key=lambda x: (x.score, x.published), reverse=True)

    max_posts = effective_max_posts_per_run()
    to_post: List[Item] = []

    # Primary first
    for it in primary_candidates:
        if max_posts is not None and len(to_post) >= max_posts:
            break
        to_post.append(it)

    # Others up to daily cap
    other_budget = max(0, OTHER_DAILY_CAP - int(daily_other.get("count", 0)))
    for it in other_candidates:
        if max_posts is not None and len(to_post) >= max_posts:
            break
        if other_budget <= 0:
            break
        to_post.append(it)
        other_budget -= 1

    # Optional per-source cap
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
        if successful_feed_fetches > 0:
            state["last_run_success_at"] = utcnow().isoformat()
        save_state(state)
        return

    sess = bsky_create_session()
    access_jwt = sess["accessJwt"]
    did = sess["did"]

    posted_any = 0
    for it in to_post:
        text = format_post_text(it)
        facets = make_link_facet(text, it.url)

        # Try to build an external embed to force a “card”
        embed = None
        try:
            card = fetch_external_card(it.url)
            embed = build_external_embed(access_jwt, card)
        except Exception:
            embed = None

        try:
            print(f"[post] ({it.score}) {it.publication}: {it.title} -> {it.url}")
            bsky_post(access_jwt, did, text, facets=facets, embed=embed)
            posted[it.url] = utcnow().isoformat()
            posted_any += 1
            if not it.is_primary:
                daily_other["count"] = int(daily_other.get("count", 0)) + 1
        except Exception as e:
            print(f"[warn] post failed: {e}")

    state["posted"] = posted
    state["daily_other"] = daily_other
    if successful_feed_fetches > 0:
        state["last_run_success_at"] = utcnow().isoformat()
    save_state(state)

    if posted_any == 0:
        raise SystemExit("All posts failed.")
    print(f"Posted {posted_any} items.")


if __name__ == "__main__":
    main()
