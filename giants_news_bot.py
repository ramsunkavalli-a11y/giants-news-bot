import argparse
import base64
import html as html_lib
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


# ============================================================
# Config
# ============================================================
TEAM_ID = 137  # SF Giants

HOURS_BACK = int(os.getenv("HOURS_BACK", "8"))
MAX_POSTS_PER_RUN = int(os.getenv("MAX_POSTS_PER_RUN", "0") or "0")  # 0 => unlimited
STATE_FILE = os.getenv("STATE_FILE", "state.json")
KEEP_POSTED_DAYS = int(os.getenv("KEEP_POSTED_DAYS", "21"))

OTHER_DAILY_CAP = int(os.getenv("OTHER_DAILY_CAP", "2"))  # non-primary cap per day

DEBUG_REJECTIONS = os.getenv("DEBUG_REJECTIONS", "0") in {"1", "true", "yes"}

# Metadata enrichment: do this only for finalists (good cards)
ENRICH_FINALISTS = os.getenv("ENRICH_FINALISTS", "true").lower() in {"1", "true", "yes"}
MAX_FINALIST_META_FETCHES = int(os.getenv("MAX_FINALIST_META_FETCHES", "12"))

# Optional thumbnail upload (safe caps)
ENABLE_THUMBS = os.getenv("ENABLE_THUMBS", "false").lower() in {"1", "true", "yes"}
MAX_THUMB_BYTES = int(os.getenv("MAX_THUMB_BYTES", str(900 * 1024)))  # <1MB default
THUMB_TIMEOUT = int(os.getenv("THUMB_TIMEOUT", "15"))

DRY_RUN = os.getenv("DRY_RUN", "0").lower() in {"1", "true", "yes"}

# Google News can be huge; cap entries
MAX_GOOGLE_ENTRIES_PER_FEED = int(os.getenv("MAX_GOOGLE_ENTRIES_PER_FEED", "25"))

# GN newer tokens sometimes need extra fetch; keep off unless you want it
ENABLE_GN_DEEP_RESOLVE = os.getenv("ENABLE_GN_DEEP_RESOLVE", "true").lower() in {"1", "true", "yes"}

UA = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0 Safari/537.36 GiantsNewsBot/4.0",
)

BSKY_IDENTIFIER = os.environ["BSKY_IDENTIFIER"]
BSKY_APP_PASSWORD = os.environ["BSKY_APP_PASSWORD"]
BSKY_PDS = os.getenv("BSKY_PDS", "https://bsky.social")


# Domains you explicitly care about (priority sources)
PRIMARY_DOMAINS = {
    "sfchronicle.com",
    "mercurynews.com",
    "nbcsportsbayarea.com",
    "sfgiants.com",
    "mlb.com",
    "sfstandard.com",
    "knbr.com",
    "sfgate.com",
    "theathletic.com",  # legacy host
    "apnews.com",
    "fangraphs.com",
    "baseballamerica.com",
    "nytimes.com",  # Athletic is often here: /athletic/
}

# Paywall tagging
PAYWALL_DOMAINS = {
    "theathletic.com",
    "sfchronicle.com",
    "mercurynews.com",
    "baseballamerica.com",
    "nytimes.com",  # for /athletic/
}

AGGREGATOR_BLOCKLIST = {
    "news.google.com",
    "feedspot.com",
    "feedly.com",
    "newsbreak.com",
    "ground.news",
}

TRACKER_BLOCKLIST = {
    "doubleclick.net",
    "googlesyndication.com",
    "adservice.google.com",
    "securepubads.g.doubleclick.net",
    "tpc.googlesyndication.com",
    "stats.g.doubleclick.net",
    "ad.doubleclick.net",
}

REFERENCE_DOMAIN_BLOCKLIST = {
    "w3.org",
    "schema.org",
    "fonts.googleapis.com",
}

# ============================================================
# Relevance knobs
# ============================================================
BASEBALL_CONTEXT_TERMS = [
    "mlb", "baseball", "spring training", "opening day", "postseason", "playoffs",
    "trade", "traded", "waivers", "claimed", "dfa", "designated for assignment",
    "optioned", "called up", "roster", "40-man", "injured list", "il",
    "pitcher", "rotation", "bullpen", "prospect", "draft", "minor league",
    "fangraphs", "baseball america", "mlb.com", "sfgiants",
]

NEGATIVE_PHRASES = [
    "dear abby",
    "new york giants", "ny giants",
    "nfl", "super bowl", "touchdown", "quarterback",
]

IMPORTANT_KEYWORDS = {
    "trade", "traded", "acquired", "waiver", "claimed", "dfa", "optioned",
    "signed", "signing", "extension", "injury", "surgery", "rehab",
    "prospect", "promotion", "bullpen", "rotation", "starter", "closer",
}

FRONT_OFFICE_POWER = {
    "Greg Johnson",
    "Rob Dean",
    "Larry Baer",
    "Buster Posey",
    "Zack Minasian",
}
KEY_PEOPLE = set(FRONT_OFFICE_POWER) | {"Tony Vitello"}


# ============================================================
# URL / Story-likeness gating (THIS is the main fix)
# ============================================================

# Domain-specific article URL signatures (strong)
ARTICLE_PATTERNS: Dict[str, List[str]] = {
    # NBCSBA articles commonly end with a big numeric ID segment
    "nbcsportsbayarea.com": [r"/\d{6,}/?$"],
    "apnews.com": [r"/article/"],
    "sfchronicle.com": [r"/20\d{2}/\d{2}/\d{2}/"],
    "mercurynews.com": [r"/20\d{2}/\d{2}/\d{2}/"],
    "knbr.com": [r"/20\d{2}/\d{2}/\d{2}/", r"/\d{6,}/?$"],
    "fangraphs.com": [r"/20\d{2}/\d{2}/\d{2}/"],
    # MLB / SFGiants news slugs
    "mlb.com": [r"^/giants/news/[^/]+/?$", r"^/news/[^/]+/?$"],
    "sfgiants.com": [r"^/news/[^/]+/?$"],
    # SFGate article URLs often have /article/… or end in .php/.html
    "sfgate.com": [r"/article/", r"\.(php|html)$"],
    # SF Standard often uses dated paths or long slugs (keep generic fallback too)
    "sfstandard.com": [r"/20\d{2}/\d{2}/\d{2}/", r"/[a-z0-9-]{20,}/?$"],
    "baseballamerica.com": [r"/stories?/", r"/prospects?/", r"/news/"],
    "nytimes.com": [r"/athletic/\d+/"],  # Athletic host
}

# Slugs/paths that are *not* articles (hubs/sections/tags) — block hard.
HUB_PATH_HINTS = {
    "/tag/", "/tags/", "/topic/", "/topics/", "/category/", "/categories/",
    "/section/", "/sections/", "/hub/",
    "/roster", "/schedule", "/tickets", "/standings", "/stats",
}

# Very common Giants hubs that look like they could be titles but aren’t stories
HUB_SLUG_BLOCKLIST = {
    "giants-news",
    "giants-rumors",
    "giants-playoffs",
    "giants-spring-training",
    "san-francisco-giants",
    "sf-giants",
    "giants",
    "news",
    "sports",
}


# ============================================================
# Data structures
# ============================================================
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
    og_image: str = ""


# ============================================================
# Regex helpers
# ============================================================
RE_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
RE_SPACE = re.compile(r"\s+")
RE_URL = re.compile(r"https?://[^\s\"'<>]+", re.I)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def norm_text(s: str) -> str:
    return RE_SPACE.sub(" ", (s or "").strip().lower())


def tokenize_words(text: str) -> Set[str]:
    return set(w.lower() for w in RE_WORD.findall(text or ""))


def looks_like_url(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and bool(RE_URL.fullmatch(t))


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
    return (not d) or d in REFERENCE_DOMAIN_BLOCKLIST or any(d.endswith("." + x) for x in REFERENCE_DOMAIN_BLOCKLIST)


def canonicalize_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u

    while u and u[-1] in ")],.;":
        u = u[:-1]

    try:
        p = urlparse(u)
        q = parse_qs(p.query, keep_blank_values=True)

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


def path_segs(url: str) -> List[str]:
    try:
        p = urlparse(url).path or ""
    except Exception:
        return []
    return [s for s in p.strip("/").split("/") if s]


def looks_like_hub(url: str) -> bool:
    try:
        p = (urlparse(url).path or "").lower()
    except Exception:
        return True
    for hint in HUB_PATH_HINTS:
        if hint in p:
            return True
    segs = path_segs(url)
    if not segs:
        return True
    last = segs[-1].lower()
    if last in HUB_SLUG_BLOCKLIST:
        return True
    # shallow paths are often hubs
    if len(segs) <= 1:
        return True
    return False


def matches_article_pattern(url: str, domain: str) -> bool:
    if not domain:
        domain = domain_of(url)
    pth = urlparse(url).path or ""
    pats = ARTICLE_PATTERNS.get(domain, [])
    for pat in pats:
        if pat.startswith("^"):
            if re.search(pat, pth):
                return True
        else:
            if re.search(pat, pth):
                return True
    return False


def generic_article_heuristic(url: str) -> bool:
    pth = (urlparse(url).path or "").lower()

    # date path => usually story
    if re.search(r"/20\d{2}/\d{1,2}/\d{1,2}/", pth):
        return True

    segs = [s for s in pth.strip("/").split("/") if s]
    if len(segs) >= 3:
        last = segs[-1]
        if re.search(r"\.(html|php)$", last):
            return True
        # long slug with hyphens is commonly a story page
        if "-" in last and len(last) >= 18 and last not in HUB_SLUG_BLOCKLIST:
            return True
    return False


def is_story_url(url: str) -> bool:
    u = canonicalize_url(url)
    d = domain_of(u)
    if not u.startswith("http") or not d:
        return False
    if is_google_host(u) or is_blocked_domain(d) or is_reference_domain(d):
        return False
    if looks_like_hub(u):
        return False
    if matches_article_pattern(u, d):
        return True
    return generic_article_heuristic(u)


def is_paywalled(url: str, domain: str) -> bool:
    d = (domain or "").lower()
    if any(d == pw or d.endswith("." + pw) for pw in PAYWALL_DOMAINS):
        # Treat NYT as paywall only for Athletic URLs
        if d == "nytimes.com":
            return "/athletic/" in (urlparse(url).path or "").lower()
        return True
    return False


# ============================================================
# State
# ============================================================
def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {
            "posted": {},
            "daily_other": {},
            "last_run_success_at": None,
            "redirect_cache": {},
        }
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)
    state.setdefault("posted", {})
    state.setdefault("daily_other", {})
    state.setdefault("last_run_success_at", None)
    state.setdefault("redirect_cache", {})
    return state


def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def prune_state(state: Dict[str, Any], keep_days: int = KEEP_POSTED_DAYS) -> None:
    cutoff = utcnow() - timedelta(days=keep_days)
    posted = state.get("posted", {})
    for url, ts in list(posted.items()):
        try:
            dt = dtparser.isoparse(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt < cutoff:
                posted.pop(url, None)
        except Exception:
            posted.pop(url, None)
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


def cached_final_url(url: str, state: Dict[str, Any], max_age_hours: int = 24) -> str:
    cache = state.setdefault("redirect_cache", {})
    now = utcnow()
    row = cache.get(url)
    if row:
        try:
            ts = dtparser.isoparse(row.get("ts", ""))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if (now - ts) < timedelta(hours=max_age_hours):
                return row.get("final", url)
        except Exception:
            pass

    final = url
    try:
        r = safe_get(url, timeout=15, retries=1)
        final = canonicalize_url(r.url or url)
    except Exception:
        final = url

    cache[url] = {"final": final, "ts": now.isoformat()}
    return final


# ============================================================
# Relevance / scoring
# ============================================================
def has_negative(text: str) -> bool:
    t = norm_text(text)
    return any(neg in t for neg in NEGATIVE_PHRASES)


def has_baseball_context(text: str) -> bool:
    t = norm_text(text)
    return any(term in t for term in BASEBALL_CONTEXT_TERMS)


def mentions_team_strong(text: str) -> bool:
    t = norm_text(text)
    return ("san francisco giants" in t) or ("sf giants" in t)


def is_allowed_item(title: str, summary: str, publication: str) -> Tuple[bool, str]:
    blob = f"{title}\n{summary}\n{publication}"
    if has_negative(blob):
        return False, "negative_phrase"
    if mentions_team_strong(blob):
        return True, "team_match"
    if not has_baseball_context(blob):
        return False, "no_baseball_context"
    # allow “Giants” alone if it’s baseball context
    if re.search(r"(?<!\w)giants(?!\w)", norm_text(blob)):
        return True, "giants_context"
    return False, "no_team_signal"


def importance_score(it: Item, echo_count: int) -> int:
    blob = f"{it.title} {it.raw_summary} {it.publication}".lower()
    score = 0

    for name in KEY_PEOPLE:
        if name.lower() in blob:
            score += 10

    if ("san francisco giants" in blob) or ("sf giants" in blob):
        score += 4

    for kw in IMPORTANT_KEYWORDS:
        if kw in blob:
            score += 2

    if echo_count >= 2:
        score += min(8, 3 * (echo_count - 1))

    hours_old = max(0.0, (utcnow() - it.published).total_seconds() / 3600.0)
    if hours_old <= 2:
        score += 3
    elif hours_old <= 6:
        score += 2
    elif hours_old <= 12:
        score += 1

    if it.is_primary:
        score += 2

    # penalize suspiciously generic titles (hubs often look like this)
    if it.title and len(it.title.split()) <= 2:
        score -= 2

    return score


def title_hash(title: str) -> str:
    t = re.sub(r"[^a-z0-9 ]+", "", (title or "").lower())
    return re.sub(r"\s+", " ", t).strip()


# ============================================================
# Google News RSS helpers
# ============================================================
def google_news_rss_url(query: str) -> str:
    return f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"


def looks_like_google_news_rss(feed_url: str) -> bool:
    u = (feed_url or "").lower()
    return "news.google.com/rss" in u


def decode_google_redirect_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u
    try:
        parsed = urlparse(u)
        host = parsed.netloc.lower()
        if host.endswith("google.com") and parsed.path.startswith("/url"):
            q = parse_qs(parsed.query)
            for key in ("url", "q", "u"):
                for val in (q.get(key) or []):
                    cand = unquote(val).strip()
                    if cand.startswith("http") and not is_google_host(cand):
                        return cand
    except Exception:
        pass
    return u


def _read_varint(buf: bytes, start: int = 0) -> Tuple[int, int]:
    """
    Read protobuf varint from buf[start:].
    Returns (value, bytes_consumed).
    """
    value = 0
    shift = 0
    i = start
    while i < len(buf):
        b = buf[i]
        value |= (b & 0x7F) << shift
        i += 1
        if not (b & 0x80):
            return value, i - start
        shift += 7
        if shift > 63:
            break
    return 0, 0


def decode_google_news_token_url(gn_url: str) -> str:
    """
    Attempt offline decoding of GN /articles/<token> style URLs.
    This covers one common token format (protobuf-ish wrapper).
    If it fails, caller should try fetch fallback.
    """
    u = (gn_url or "").strip()
    if not u:
        return ""
    try:
        p = urlparse(u)
        if "news.google.com" not in (p.netloc or ""):
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

        # Strip known prefix/suffix seen in the format
        if raw.startswith(b"\x08\x13\x22"):
            raw = raw[3:]
        if raw.endswith(b"\xd2\x01\x00"):
            raw = raw[:-3]

        ln, used = _read_varint(raw, 0)
        if ln <= 0 or used <= 0:
            # fallback: try scan bytes for http
            txt = raw.decode("utf-8", errors="ignore")
            m = re.search(r"https?://[^\s\"'<>\\]+", txt)
            if not m:
                return ""
            cand = canonicalize_url(m.group(0))
            return cand if is_story_url(cand) else ""

        start = used
        end = start + ln
        if end > len(raw):
            return ""

        cand = raw[start:end].decode("utf-8", errors="ignore").strip()
        cand = canonicalize_url(cand)
        if cand.startswith("http") and is_story_url(cand):
            return cand
    except Exception:
        return ""
    return ""


def resolve_google_news_article(gn_url: str) -> str:
    """
    Resolve a GN URL to a publisher URL:
      - decode google.com/url?url=
      - try offline token decode
      - optionally fetch GN page and scan for outbound URL
    """
    u = (gn_url or "").strip()
    if not u:
        return ""

    u2 = decode_google_redirect_url(u)
    if u2 != u and u2.startswith("http") and is_story_url(u2):
        return canonicalize_url(u2)

    tok = decode_google_news_token_url(u)
    if tok:
        return tok

    if not ENABLE_GN_DEEP_RESOLVE:
        return ""

    # Fetch fallback: follow redirects & scan HTML for external URL
    try:
        r = safe_get(u, timeout=20, retries=1)
        final = (r.url or "").strip()
        final_dec = decode_google_redirect_url(final)
        if final_dec.startswith("http") and is_story_url(final_dec):
            return canonicalize_url(final_dec)

        html = r.text or ""

        # Look for explicit google redirect URLs in page source
        m = re.search(r"https?://www\.google\.com/url\?[^\"'<> ]+", html, flags=re.I)
        if m:
            cand = decode_google_redirect_url(m.group(0))
            if cand.startswith("http") and is_story_url(cand):
                return canonicalize_url(cand)

        # Some pages contain data-n-au outbound URL
        m2 = re.search(r'data-n-au="([^"]+)"', html, flags=re.I)
        if m2:
            cand = unquote(m2.group(1)).strip()
            if cand.startswith("http") and is_story_url(cand):
                return canonicalize_url(cand)

        # Last resort: scan for any http(s) that looks like a story URL
        for uu in RE_URL.findall(html):
            cand = decode_google_redirect_url(uu.strip())
            if cand.startswith("http") and is_story_url(cand):
                return canonicalize_url(cand)
    except Exception:
        return ""
    return ""


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


def extract_author_from_entry(entry: Any) -> str:
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


def fetch_rss_items(feed_url: str, source_label: str, cutoff: datetime) -> Tuple[List[Item], Dict[str, int]]:
    r = safe_get(feed_url, timeout=30)
    r.raise_for_status()
    fp = feedparser.parse(r.text)

    items: List[Item] = []
    total_entries = len(fp.entries)
    after_cutoff = 0
    after_url_ok = 0
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
            continue

        headline, pub_from_title = extract_publication_from_title(raw_title)
        publication = pub_from_title or source_label
        author = extract_author_from_entry(e)

        url = ""
        if is_gn:
            link = (e.get("link") or "").strip()
            if link:
                url = resolve_google_news_article(link) or canonicalize_url(link)
        else:
            url = canonicalize_url((e.get("link") or "").strip())

        if not url:
            if DEBUG_REJECTIONS:
                print(f"[debug][reject][{source_label}] no_url title={headline[:90]}")
            continue

        url = canonicalize_url(url)
        d = domain_of(url)
        if not d or is_google_host(url) or is_blocked_domain(d) or is_reference_domain(d):
            if DEBUG_REJECTIONS:
                print(f"[debug][reject][{source_label}] blocked_domain domain={d} url={url[:120]}")
            continue

        # CRITICAL: enforce story URL globally
        if not is_story_url(url):
            if DEBUG_REJECTIONS:
                print(f"[debug][reject][{source_label}] not_story_url url={url[:140]} title={headline[:90]}")
            continue
        after_url_ok += 1

        summary = e.get("summary", "") or e.get("description", "") or ""
        allowed, reason = is_allowed_item(headline, summary, publication)
        if not allowed:
            if DEBUG_REJECTIONS:
                print(f"[debug][reject][{source_label}] {reason} title={headline[:90]}")
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
        "after_url_ok": after_url_ok,
        "after_relevance": after_relevance,
    }
    return items, metrics


# ============================================================
# Sitemap ingestion (non-RSS path, but still URL-gated)
# ============================================================
@dataclass
class SitemapSource:
    name: str
    domain: str
    allow_path_substrings: List[str]  # cheap prefilter, before story gating


def discover_sitemaps(domain: str) -> List[str]:
    base = f"https://{domain}"
    found: List[str] = []

    # robots.txt (Sitemap:)
    try:
        rr = safe_get(urljoin(base, "/robots.txt"), timeout=12, retries=1)
        if rr.status_code < 400:
            for line in (rr.text or "").splitlines():
                if line.lower().startswith("sitemap:"):
                    u = line.split(":", 1)[1].strip()
                    if u:
                        found.append(canonicalize_url(u))
    except Exception:
        pass

    # common defaults
    found.append(urljoin(base, "/sitemap.xml"))
    found.append(urljoin(base, "/news-sitemap.xml"))

    # de-dupe
    out: List[str] = []
    seen: Set[str] = set()
    for u in found:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def parse_sitemap_xml(xml: str) -> Tuple[List[str], List[Tuple[str, Optional[datetime]]]]:
    sitemap_urls: List[str] = []
    story_urls: List[Tuple[str, Optional[datetime]]] = []
    if not xml:
        return sitemap_urls, story_urls

    # quick loc scan
    for loc in re.findall(r"<loc>(.*?)</loc>", xml, flags=re.I | re.S):
        u = html_lib.unescape((loc or "").strip())
        if not u:
            continue
        if re.search(r"sitemap", u, flags=re.I):
            sitemap_urls.append(u)

    # url blocks with optional lastmod
    for block in re.findall(r"<url>(.*?)</url>", xml, flags=re.I | re.S):
        mloc = re.search(r"<loc>(.*?)</loc>", block, flags=re.I | re.S)
        if not mloc:
            continue
        loc = html_lib.unescape(mloc.group(1).strip())
        lm = None
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


def fetch_sitemap_candidates(src: SitemapSource, cutoff: datetime, state: Dict[str, Any]) -> List[Item]:
    """
    Pull recent-ish URLs from sitemaps, prefilter by path substring, then hard story-gate.
    Titles will be enriched later for finalists.
    """
    queue = discover_sitemaps(src.domain)
    visited: Set[str] = set()
    collected: List[Tuple[str, Optional[datetime]]] = []

    while queue and len(visited) < 20 and len(collected) < 1500:
        sm = queue.pop(0)
        if sm in visited:
            continue
        visited.add(sm)
        try:
            r = safe_get(sm, timeout=15, retries=1)
            if r.status_code >= 400:
                continue
            child_maps, urls = parse_sitemap_xml(r.text or "")
            # prioritize news sitemaps
            child_maps.sort(key=lambda x: 0 if "news" in x.lower() else 1)
            for c in child_maps:
                if c not in visited and c not in queue and len(queue) < 50:
                    queue.append(c)
            collected.extend(urls)
        except Exception:
            continue

    # Reduce: newest first by lastmod (unknown lastmod treated as old-ish)
    def key_lm(row: Tuple[str, Optional[datetime]]) -> datetime:
        return row[1] or datetime(1970, 1, 1, tzinfo=timezone.utc)

    collected.sort(key=key_lm, reverse=True)

    out: List[Item] = []
    posted = state.get("posted", {})
    seen_local: Set[str] = set()

    for u, lm in collected[:700]:
        u = canonicalize_url(u)
        if not u.startswith("http"):
            continue
        if u in posted:
            continue
        if u in seen_local:
            continue

        # cheap allow-list filter (avoid crawling whole site)
        pth = (urlparse(u).path or "").lower()
        if src.allow_path_substrings and not any(sub.lower() in pth for sub in src.allow_path_substrings):
            continue

        # redirect cache (stabilize)
        u = cached_final_url(u, state)
        u = canonicalize_url(u)

        if u in seen_local:
            continue
        seen_local.add(u)

        d = domain_of(u)
        if not d or is_blocked_domain(d) or is_reference_domain(d) or is_google_host(u):
            continue

        # hard story gate
        if not is_story_url(u):
            if DEBUG_REJECTIONS:
                print(f"[debug][reject][sitemap:{src.name}] not_story_url {u}")
            continue

        published = lm or utcnow()
        if published < cutoff:
            continue

        is_primary = any(d == pd or d.endswith("." + pd) for pd in PRIMARY_DOMAINS)
        out.append(
            Item(
                title="",  # enrich later
                url=u,
                publication=src.name,
                published=published,
                domain=d,
                is_primary=is_primary,
                raw_summary="",
            )
        )

    return out


# ============================================================
# HTML metadata extraction (for good cards + to replace slug titles)
# ============================================================
class MetaParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.canonical = ""
        self.og_url = ""
        self.title = ""
        self.og_title = ""
        self.description = ""
        self.og_description = ""
        self.author = ""
        self.published = ""
        self.og_image = ""
        self._in_title = False
        self._in_jsonld = False
        self._jsonld_buf: List[str] = []
        self.jsonld_blobs: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        attr = {k.lower(): (v or "") for k, v in attrs}
        t = tag.lower()

        if t == "link" and attr.get("rel", "").lower() == "canonical":
            href = attr.get("href", "").strip()
            if href:
                self.canonical = urljoin(self.base_url, href)

        if t == "meta":
            prop = attr.get("property", "").lower()
            name = attr.get("name", "").lower()
            content = (attr.get("content") or "").strip()

            if prop == "og:url" and content:
                self.og_url = urljoin(self.base_url, content)
            if prop == "og:title" and content:
                self.og_title = content
            if prop == "og:description" and content:
                self.og_description = content
            if prop == "og:image" and content:
                self.og_image = urljoin(self.base_url, content)

            if name in {"author", "parsely-author", "article:author", "byl"} and content:
                self.author = content

            if prop in {"article:published_time", "og:updated_time"} and content:
                self.published = content
            if name in {"pubdate", "publishdate", "date", "dc.date", "dc.date.issued"} and content and not self.published:
                self.published = content

            if name == "description" and content and not self.description:
                self.description = content

        if t == "title":
            self._in_title = True

        if t == "script":
            if (attr.get("type", "") or "").lower() == "application/ld+json":
                self._in_jsonld = True
                self._jsonld_buf = []

    def handle_endtag(self, tag: str):
        t = tag.lower()
        if t == "title":
            self._in_title = False
        if t == "script" and self._in_jsonld:
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
    out = {"title": "", "author": "", "published": "", "description": "", "canonical": "", "image": ""}
    for b in blobs:
        try:
            obj = json.loads(b)
        except Exception:
            continue

        candidates = []
        if isinstance(obj, list):
            candidates = obj
        elif isinstance(obj, dict):
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
            if t not in {"newsarticle", "article", "blogposting", "reportage"} and "article" not in t:
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
                elif isinstance(author, list) and author:
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

            img = c.get("image")
            if img and not out["image"]:
                if isinstance(img, str):
                    out["image"] = img.strip()
                elif isinstance(img, dict):
                    out["image"] = str(img.get("url") or "").strip()
                elif isinstance(img, list) and img:
                    out["image"] = str(img[0]).strip()

        if out["title"] or out["canonical"] or out["published"]:
            return out
    return out


def clean_description(s: str, max_len: int = 240) -> str:
    txt = html_lib.unescape(s or "")
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = RE_SPACE.sub(" ", txt).strip()
    if len(txt) > max_len:
        txt = txt[: max_len - 1].rstrip() + "…"
    return txt


def fetch_article_metadata(url: str) -> Dict[str, str]:
    """
    Fetch page HTML and extract canonical/title/description/author/published/og:image.
    """
    meta = {"canonical": "", "title": "", "description": "", "author": "", "published": "", "image": ""}
    r = safe_get(url, timeout=20, retries=1)
    r.raise_for_status()
    html = r.text or ""

    p = MetaParser(base_url=url)
    try:
        p.feed(html)
    except Exception:
        pass

    j = _extract_from_jsonld(p.jsonld_blobs)

    meta["canonical"] = (p.canonical or p.og_url or j.get("canonical") or "").strip()
    meta["title"] = (p.og_title or j.get("title") or p.title or "").strip()
    meta["description"] = (p.og_description or j.get("description") or p.description or "").strip()
    meta["author"] = (p.author or j.get("author") or "").strip()
    meta["published"] = (p.published or j.get("published") or "").strip()
    meta["image"] = (p.og_image or j.get("image") or "").strip()

    return meta


def title_from_url(url: str) -> str:
    try:
        path = (urlparse(url).path or "").strip("/")
        parts = [seg for seg in path.split("/") if seg]
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
        slug = re.sub(r"\.[A-Za-z0-9]+$", "", slug)
        slug = re.sub(r"[-_]+", " ", slug)
        slug = RE_SPACE.sub(" ", slug).strip()
        if not slug:
            return url
        return slug[:1].upper() + slug[1:]
    except Exception:
        return url


# ============================================================
# Bluesky posting
# ============================================================
def bsky_create_session() -> Dict[str, Any]:
    r = requests.post(
        f"{BSKY_PDS}/xrpc/com.atproto.server.createSession",
        json={"identifier": BSKY_IDENTIFIER, "password": BSKY_APP_PASSWORD},
        timeout=20,
        headers={"User-Agent": UA},
    )
    r.raise_for_status()
    return r.json()


def bsky_upload_blob(access_jwt: str, did: str, data: bytes, mime: str) -> Optional[Dict[str, Any]]:
    """
    Upload blob for thumbnail. Returns blob ref dict (or None).
    """
    try:
        r = requests.post(
            f"{BSKY_PDS}/xrpc/com.atproto.repo.uploadBlob",
            data=data,
            timeout=20,
            headers={
                "Authorization": f"Bearer {access_jwt}",
                "User-Agent": UA,
                "Content-Type": mime,
            },
        )
        if r.status_code >= 400:
            if DEBUG_REJECTIONS:
                print(f"[debug][thumb] uploadBlob failed {r.status_code}: {r.text[:300]}")
            return None
        js = r.json()
        return js.get("blob")
    except Exception as e:
        if DEBUG_REJECTIONS:
            print(f"[debug][thumb] upload exception: {e}")
        return None


def fetch_thumb_bytes(url: str) -> Tuple[Optional[bytes], str]:
    """
    Download og:image with size cap. Returns (bytes, mime) or (None, "").
    """
    if not url:
        return None, ""
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=THUMB_TIMEOUT, stream=True, allow_redirects=True)
        if r.status_code >= 400:
            return None, ""
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        clen = r.headers.get("Content-Length")
        if clen:
            try:
                if int(clen) > MAX_THUMB_BYTES:
                    return None, ""
            except Exception:
                pass
        if not ctype.startswith("image/"):
            return None, ""
        buf = bytearray()
        for chunk in r.iter_content(chunk_size=64 * 1024):
            if not chunk:
                break
            buf.extend(chunk)
            if len(buf) > MAX_THUMB_BYTES:
                return None, ""
        return bytes(buf), ctype
    except Exception:
        return None, ""


def build_display_line(it: Item) -> str:
    label = (it.author or "").strip() or (it.publication or "").strip()
    t = (it.title or "").strip()

    if looks_like_url(t) or not t:
        t = title_from_url(it.url)

    if is_paywalled(it.url, it.domain):
        t = f"{t} ($)"

    if label:
        return f"{label}: {t}"
    return t


def build_post_text(it: Item, max_len: int = 300) -> str:
    line = build_display_line(it)
    if len(line) <= max_len:
        return line
    return line[: max_len - 1].rstrip() + "…"


def build_external_embed(it: Item, thumb_blob: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    title = (it.title or "").strip() or title_from_url(it.url)
    desc = clean_description(it.raw_summary or "", max_len=240)
    if not desc:
        desc = (it.publication or "").strip()

    external: Dict[str, Any] = {
        "uri": it.url,
        "title": title[:300],
        "description": desc[:1000],
    }
    if thumb_blob:
        external["thumb"] = thumb_blob

    return {
        "$type": "app.bsky.embed.external",
        "external": external,
    }


def bsky_post(access_jwt: str, did: str, text: str, embed: Dict[str, Any]) -> None:
    record: Dict[str, Any] = {
        "$type": "app.bsky.feed.post",
        "text": text,
        "createdAt": utcnow().isoformat().replace("+00:00", "Z"),
        "embed": embed,
    }
    r = requests.post(
        f"{BSKY_PDS}/xrpc/com.atproto.repo.createRecord",
        json={"repo": did, "collection": "app.bsky.feed.post", "record": record},
        timeout=20,
        headers={"Authorization": f"Bearer {access_jwt}", "User-Agent": UA},
    )
    if r.status_code >= 400:
        print(f"[error] createRecord {r.status_code}: {r.text[:1200]}")
    r.raise_for_status()


# ============================================================
# Main orchestration
# ============================================================
def effective_max_posts_per_run() -> Optional[int]:
    return None if MAX_POSTS_PER_RUN <= 0 else MAX_POSTS_PER_RUN


def main(dry_run: bool = DRY_RUN) -> None:
    state = load_state()
    prune_state(state)
    daily_other = get_daily_other_counter(state)
    cutoff = compute_cutoff(state)
    print(f"[info] cutoff={cutoff.isoformat()} (hours_back={HOURS_BACK})")

    # ------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------
    rss_feeds: List[Tuple[str, str]] = [
        ("SF Standard", "https://sfstandard.com/sports/feed"),
        ("SFGate", "https://www.sfgate.com/sports/feed/san-francisco-giants-rss-feed-428.php"),
    ]

    domain_queries = [
        ("SF Chronicle", '(("San Francisco Giants" OR "SF Giants" OR Giants) AND (MLB OR baseball)) site:sfchronicle.com'),
        ("Mercury News", '(("San Francisco Giants" OR "SF Giants" OR Giants) AND (MLB OR baseball)) site:mercurynews.com'),
        ("NBC Sports Bay Area", '(("San Francisco Giants" OR "SF Giants" OR Giants) AND (MLB OR baseball)) site:nbcsportsbayarea.com'),
        ("The Athletic", '(("San Francisco Giants" OR "SF Giants" OR Giants) AND (MLB OR baseball)) (site:theathletic.com OR site:nytimes.com/athletic)'),
        ("Associated Press", '(("San Francisco Giants" OR "SF Giants" OR Giants) AND (MLB OR baseball)) site:apnews.com'),
        ("SFGiants.com / MLB Giants", '(("San Francisco Giants" OR "SF Giants" OR Giants) AND (MLB OR baseball)) (site:mlb.com/giants OR site:sfgiants.com)'),
        ("FanGraphs", '(("San Francisco Giants" OR "SF Giants" OR Giants) AND (MLB OR baseball)) site:fangraphs.com'),
        ("Baseball America", '(("San Francisco Giants" OR "SF Giants" OR Giants) AND (MLB OR baseball)) site:baseballamerica.com'),
        ("KNBR", '(("San Francisco Giants" OR "SF Giants" OR Giants) AND (MLB OR baseball)) site:knbr.com'),
    ]
    for name, q in domain_queries:
        rss_feeds.append((f"Google News: {name}", google_news_rss_url(q)))
    rss_feeds.append(("Google News: Broad", google_news_rss_url('(("San Francisco Giants" OR "SF Giants") AND (MLB OR baseball))')))

    sitemap_sources: List[SitemapSource] = [
        SitemapSource("SF Chronicle", "sfchronicle.com", ["/sports/giants/"]),
        SitemapSource("Mercury News", "mercurynews.com", ["/tag/san-francisco-giants/", "/sports/"]),
        SitemapSource("NBC Sports Bay Area", "nbcsportsbayarea.com", ["/mlb/san-francisco-giants/"]),
        SitemapSource("AP News", "apnews.com", ["/hub/san-francisco-giants", "/article/"]),
        SitemapSource("KNBR", "knbr.com", ["/category/san-francisco-giants/", "/20", "/giants"]),
        SitemapSource("FanGraphs", "fangraphs.com", ["/category/giants/", "/20"]),
        SitemapSource("Baseball America", "baseballamerica.com", ["/teams/mlb/san-francisco-giants/", "/stories/", "/news/"]),
        SitemapSource("MLB Giants News", "mlb.com", ["/giants/news/"]),
        SitemapSource("SFGiants.com News", "sfgiants.com", ["/news/"]),
    ]

    # ------------------------------------------------------------
    # Collect candidates
    # ------------------------------------------------------------
    all_items: List[Item] = []
    successful_sources = 0

    for label, feed_url in rss_feeds:
        try:
            items, metrics = fetch_rss_items(feed_url, label, cutoff)
            successful_sources += 1
            print(
                "[info] rss metrics"
                f" source={label}"
                f" total={metrics['total_entries']}"
                f" after_cutoff={metrics['after_cutoff']}"
                f" after_url_ok={metrics['after_url_ok']}"
                f" after_relevance={metrics['after_relevance']}"
            )
            all_items.extend(items)
        except Exception as ex:
            print(f"[warn] rss feed failed: {label}: {ex}")

    for src in sitemap_sources:
        try:
            items = fetch_sitemap_candidates(src, cutoff, state)
            successful_sources += 1
            print(f"[info] sitemap metrics source={src.name} accepted_urls={len(items)}")
            all_items.extend(items)
        except Exception as ex:
            print(f"[warn] sitemap source failed: {src.name}: {ex}")

    if not all_items:
        print("[info] No eligible items found.")
        if successful_sources > 0:
            state["last_run_success_at"] = utcnow().isoformat()
        save_state(state)
        return

    # ------------------------------------------------------------
    # Dedupe (URL + normalize)
    # ------------------------------------------------------------
    posted = state.get("posted", {})
    seen: Set[str] = set()
    deduped: List[Item] = []
    for it in all_items:
        it.url = canonicalize_url(it.url)
        it.domain = domain_of(it.url)
        if not it.domain or it.url in seen:
            continue
        seen.add(it.url)
        if it.url in posted:
            continue
        it.is_primary = any(it.domain == d or it.domain.endswith("." + d) for d in PRIMARY_DOMAINS)
        deduped.append(it)
    all_items = deduped

    # ------------------------------------------------------------
    # Enrich a *small* subset early if they have empty title (sitemap items)
    # so scoring isn't garbage.
    # ------------------------------------------------------------
    if ENRICH_FINALISTS:
        # Pick candidates most likely to be posted: primary + newest + missing titles
        want = sorted(all_items, key=lambda x: (x.is_primary, x.published), reverse=True)
        budget = MAX_FINALIST_META_FETCHES

        for it in want:
            if budget <= 0:
                break
            if it.title and len(it.title.strip()) >= 8 and it.raw_summary:
                continue
            try:
                meta = fetch_article_metadata(it.url)
                budget -= 1

                # canonical swap
                can = canonicalize_url(meta.get("canonical") or "")
                if can and can.startswith("http") and domain_of(can) and is_story_url(can):
                    it.url = can
                    it.domain = domain_of(can)

                # apply best title/desc/author
                mt = (meta.get("title") or "").strip()
                md = (meta.get("description") or "").strip()
                ma = (meta.get("author") or "").strip()
                mi = (meta.get("image") or "").strip()

                if mt and len(mt) >= 8:
                    it.title = mt
                if md and len(md) >= 20:
                    it.raw_summary = md
                if ma and len(ma) <= 80:
                    it.author = ma
                if mi:
                    it.og_image = mi

                # published time if parsable
                pr = (meta.get("published") or "").strip()
                if pr:
                    try:
                        pdt = dtparser.parse(pr)
                        if pdt.tzinfo is None:
                            pdt = pdt.replace(tzinfo=timezone.utc)
                        it.published = pdt.astimezone(timezone.utc)
                    except Exception:
                        pass
            except Exception as e:
                if DEBUG_REJECTIONS:
                    print(f"[debug][meta] failed {it.url}: {e}")

    # ------------------------------------------------------------
    # Filter by relevance (after some enrichment)
    # ------------------------------------------------------------
    filtered: List[Item] = []
    for it in all_items:
        allowed, reason = is_allowed_item(it.title or title_from_url(it.url), it.raw_summary or "", it.publication)
        if not allowed:
            if DEBUG_REJECTIONS:
                print(f"[debug][reject][final_filter] {reason} url={it.url}")
            continue
        filtered.append(it)
    all_items = filtered

    if not all_items:
        print("[info] No items survived relevance filter.")
        if successful_sources > 0:
            state["last_run_success_at"] = utcnow().isoformat()
        save_state(state)
        return

    # ------------------------------------------------------------
    # Echo counts + scoring
    # ------------------------------------------------------------
    echo: Dict[str, int] = {}
    for it in all_items:
        h = title_hash(it.title or title_from_url(it.url))
        echo[h] = echo.get(h, 0) + 1
    for it in all_items:
        it.score = importance_score(it, echo.get(title_hash(it.title or title_from_url(it.url)), 1))

    # newest first for primaries, score+newness for others
    all_items.sort(key=lambda x: x.published, reverse=True)

    primary = [x for x in all_items if x.is_primary]
    other = [x for x in all_items if not x.is_primary]
    other.sort(key=lambda x: (x.score, x.published), reverse=True)

    max_posts = effective_max_posts_per_run()
    to_post: List[Item] = []

    for it in primary:
        if max_posts is not None and len(to_post) >= max_posts:
            break
        to_post.append(it)

    other_budget = max(0, OTHER_DAILY_CAP - int(daily_other.get("count", 0)))
    for it in other:
        if max_posts is not None and len(to_post) >= max_posts:
            break
        if other_budget <= 0:
            break
        to_post.append(it)
        other_budget -= 1

    if not to_post:
        print("[info] No new items to post (after caps).")
        if successful_sources > 0:
            state["last_run_success_at"] = utcnow().isoformat()
        save_state(state)
        return

    if dry_run:
        print("[dry-run] items to post:")
        for it in to_post:
            print(f"[dry-run][{it.score}] {it.publication} | {it.title or title_from_url(it.url)} | {it.url}")
        if successful_sources > 0:
            state["last_run_success_at"] = utcnow().isoformat()
        save_state(state)
        return

    # ------------------------------------------------------------
    # Post to Bluesky
    # ------------------------------------------------------------
    sess = bsky_create_session()
    access_jwt = sess["accessJwt"]
    did = sess["did"]

    posted_any = 0
    for it in to_post:
        # Final sanity: story-gate again after any canonical/redirect
        it.url = canonicalize_url(it.url)
        it.domain = domain_of(it.url)
        if not is_story_url(it.url):
            print(f"[warn] skipping non-story at post time: {it.url}")
            continue

        # Optional thumb
        thumb_blob = None
        if ENABLE_THUMBS and it.og_image:
            img_bytes, mime = fetch_thumb_bytes(it.og_image)
            if img_bytes and mime:
                thumb_blob = bsky_upload_blob(access_jwt, did, img_bytes, mime)

        text = build_post_text(it)
        embed = build_external_embed(it, thumb_blob)

        try:
            print(f"[post] ({it.score}) {it.publication}: {it.title or title_from_url(it.url)} -> {it.url}")
            bsky_post(access_jwt, did, text, embed)
            state["posted"][it.url] = utcnow().isoformat()
            posted_any += 1
            if not it.is_primary:
                daily_other["count"] = int(daily_other.get("count", 0)) + 1
        except Exception as e:
            print(f"[warn] post failed: {e}")

    state["daily_other"] = daily_other
    if successful_sources > 0:
        state["last_run_success_at"] = utcnow().isoformat()
    save_state(state)

    if posted_any == 0:
        raise SystemExit("All posts failed.")
    print(f"Posted {posted_any} items.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SF Giants Bluesky news bot")
    ap.add_argument("--dry-run", action="store_true", help="Run without posting")
    args = ap.parse_args()
    main(dry_run=args.dry_run or DRY_RUN)
