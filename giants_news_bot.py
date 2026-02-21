import json
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


# -----------------------------
# Time / text helpers
# -----------------------------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def norm_text(s: str) -> str:
    return RE_SPACE.sub(" ", (s or "").strip().lower())


def tokenize_words(text: str) -> Set[str]:
    return set(w.lower() for w in RE_WORD.findall(text or ""))


def safe_get(url: str, timeout: int = 25) -> requests.Response:
    return requests.get(url, timeout=timeout, headers={"User-Agent": UA}, allow_redirects=True)


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
    if d.endswith("google.com") or d.endswith("googleusercontent.com") or d.endswith("gstatic.com"):
        return True
    return False


def canonicalize_url(url: str) -> str:
    """
    Remove obvious tracking params, fragments, normalize.
    This improves dedupe and reduces Google/UTM noise.
    """
    u = (url or "").strip()
    if not u:
        return u
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


# -----------------------------
# URL resolution: Google News RSS article links
# -----------------------------
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
        m = re.search(r"https?://www\.google\.com/url\?[^\"\'<> ]+", html, flags=re.I)
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
            if cand.startswith("http") and d and not is_google_host(cand) and not is_blocked_domain(d):
                return canonicalize_url(cand)

    except Exception:
        pass

    return ""


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
            gn_link = (e.get("link") or "").strip()
            url = resolve_google_news_article(gn_link)
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


def fetch_listing_items(
    src: ListingSource,
    cutoff: datetime,
    full_names: Set[str],
    last_map: Dict[str, Set[str]],
    meta_fetch_budget: Dict[str, int],
) -> Tuple[List[Item], Dict[str, int]]:
    """
    Strategy:
      - Fetch listing HTML
      - Extract links
      - Filter to plausible article links
      - Fetch each candidate article's HTML metadata (limited by budget)
      - Use metadata title/author/published if present; fallback published=utcnow()
    """
    items: List[Item] = []
    total_links = 0
    kept_links = 0
    meta_attempts = 0
    meta_success = 0

    try:
        r = safe_get(src.url, timeout=25)
        r.raise_for_status()
        html = r.text or ""
    except Exception as e:
        return [], {"total_links": 0, "kept_links": 0, "meta_attempts": 0, "meta_success": 0, "error": 1}

    links = extract_links_from_listing(src.url, html)
    total_links = len(links)

    # Filter + dedupe, cap
    candidates = []
    seen = set()
    for u in links:
        u2 = canonicalize_url(u)
        if u2 in seen:
            continue
        seen.add(u2)
        if link_allowed_for_listing(src, u2):
            candidates.append(u2)
        if len(candidates) >= MAX_LISTING_LINKS_PER_SOURCE:
            break

    kept_links = len(candidates)

    for u in candidates:
        if meta_fetch_budget["remaining"] <= 0:
            break
        meta_fetch_budget["remaining"] -= 1
        meta_attempts += 1

        try:
            rr = safe_get(u, timeout=20)
            rr.raise_for_status()
            meta = parse_html_metadata(u, rr.text or "")
        except Exception:
            continue

        meta_success += 1

        canonical = canonicalize_url(meta.get("canonical") or u)
        d = domain_of(canonical)
        if not d or is_google_host(canonical) or is_blocked_domain(d):
            continue

        title = (meta.get("title") or "").strip()
        author = (meta.get("author") or "").strip()
        desc = (meta.get("description") or "").strip()

        # Published time best-effort
        published = None
        pub_raw = (meta.get("published") or "").strip()
        if pub_raw:
            try:
                dt = dtparser.parse(pub_raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                published = dt.astimezone(timezone.utc)
            except Exception:
                published = None

        if not published:
            # Without a published time, treat as “now” but still respect cutoff lightly:
            published = utcnow()

        # If the listing page includes older evergreen links, drop clearly old items.
        if published < cutoff:
            continue

        publication = src.name

        # Relevance check
        allowed, _reason = is_allowed_item(title or canonical, desc, publication, d, full_names, last_map)
        if not allowed:
            continue

        is_primary = any(d == pd or d.endswith("." + pd) for pd in PRIMARY_DOMAINS)

        items.append(
            Item(
                title=title or canonical,
                url=canonical,
                publication=publication,
                published=published,
                domain=d,
                is_primary=is_primary,
                author=author,
                raw_summary=desc,
            )
        )

    metrics = {
        "total_links": total_links,
        "kept_links": kept_links,
        "meta_attempts": meta_attempts,
        "meta_success": meta_success,
        "error": 0,
    }
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

    if is_paywalled_domain(it.domain):
        t = f"{t} ($)"

    if label:
        return f"{label}: {t}"
    return t


def clean_description(s: str, max_len: int = 240) -> str:
    txt = re.sub(r"<[^>]+>", " ", s or "")
    txt = RE_SPACE.sub(" ", txt).strip()
    if len(txt) > max_len:
        txt = txt[: max_len - 1].rstrip() + "…"
    return txt


def build_post_text(it: Item) -> str:
    line = build_display_line(it)
    u = (it.url or "").strip()
    text = f"{line}\n\n{u}".strip()
    # Keep within 300 chars
    if len(text) <= 300:
        return text

    # Truncate line to fit
    room_for_line = max(20, 300 - (len(u) + 2))
    if len(line) > room_for_line:
        line = line[: room_for_line - 1].rstrip() + "…"
    return f"{line}\n\n{u}"


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

    # Non-RSS listing pages (the “radical” part).
    # Patterns are intentionally conservative to avoid pulling unrelated site links.
    listing_sources: List[ListingSource] = [
        ListingSource(
            name="SF Chronicle (listing)",
            url="https://www.sfchronicle.com/sports/giants/",
            domain="sfchronicle.com",
            allow_patterns=["/sports/giants/"],
        ),
        # If you find a better Mercury News Giants landing page, swap it in.
        ListingSource(
            name="Mercury News (listing)",
            url="https://www.mercurynews.com/tag/san-francisco-giants/",
            domain="mercurynews.com",
            allow_patterns=["/tag/san-francisco-giants/", "re:/\\d{4}/\\d{2}/\\d{2}/"],
        ),
        ListingSource(
            name="AP News (listing)",
            url="https://apnews.com/hub/san-francisco-giants",
            domain="apnews.com",
            allow_patterns=["/article/", "/hub/san-francisco-giants"],
        ),
        ListingSource(
            name="NBCS Bay Area (listing)",
            url="https://www.nbcsportsbayarea.com/mlb/san-francisco-giants/",
            domain="nbcsportsbayarea.com",
            allow_patterns=["/mlb/san-francisco-giants/"],
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

    # Listing path (bounded by metadata fetch budget)
    meta_budget = {"remaining": MAX_META_FETCHES_PER_RUN if ENRICH_ARTICLE_METADATA else 0}
    for src in listing_sources:
        try:
            items, metrics = fetch_listing_items(src, cutoff, full_names, last_map, meta_budget)
            successful_sources += 1
            print(
                "[info] listing metrics"
                f" source={src.name}"
                f" total_links={metrics.get('total_links', 0)}"
                f" kept_links={metrics.get('kept_links', 0)}"
                f" meta_attempts={metrics.get('meta_attempts', 0)}"
                f" meta_success={metrics.get('meta_success', 0)}"
                f" budget_left={meta_budget['remaining']}"
            )
            all_items.extend(items)
        except Exception as ex:
            print(f"[warn] listing source failed: {src.name}: {ex}")

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
    main()
