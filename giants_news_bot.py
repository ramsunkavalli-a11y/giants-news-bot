import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Set, Tuple
from urllib.parse import quote, urlparse

import feedparser
import requests
from dateutil import parser as dtparser


# -----------------------------
# Config
# -----------------------------
TEAM_ID = 137  # SF Giants (MLB Stats API)

HOURS_BACK = int(os.getenv("HOURS_BACK", "8"))
MAX_POSTS_PER_RUN = int(os.getenv("MAX_POSTS_PER_RUN", "5"))
STATE_FILE = os.getenv("STATE_FILE", "state.json")

# Per-run diversity: cap how many posts from the same publication in a single run.
PER_SOURCE_CAP = int(os.getenv("PER_SOURCE_CAP", "2"))

# “Best of the rest” (outside your chosen domains): cap posts per day
OTHER_DAILY_CAP = int(os.getenv("OTHER_DAILY_CAP", "2"))

# Cache lifetimes
ROSTER_CACHE_HOURS = int(os.getenv("ROSTER_CACHE_HOURS", "24"))
STAFF_CACHE_HOURS = int(os.getenv("STAFF_CACHE_HOURS", "24"))
KEEP_POSTED_DAYS = int(os.getenv("KEEP_POSTED_DAYS", "21"))

BSKY_IDENTIFIER = os.environ["BSKY_IDENTIFIER"]  # handle or email
BSKY_APP_PASSWORD = os.environ["BSKY_APP_PASSWORD"]
BSKY_PDS = os.getenv("BSKY_PDS", "https://bsky.social")

UA = "GiantsNewsBot/2.0 (+github-actions)"


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

# Known aggregator / meta domains you likely don’t want as final links
AGGREGATOR_BLOCKLIST = {
    "news.google.com",
    "feedspot.com",
    "feedly.com",
    "newsbreak.com",
    "ground.news",
}


# -----------------------------
# Always-relevant “power” names (front office / decision makers)
# plus your explicitly requested names.
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
KEY_PEOPLE = set(FRONT_OFFICE_POWER) | {
    "Tony Vitello",
}


# -----------------------------
# Relevance rules (tight)
# -----------------------------
BASEBALL_CONTEXT_TERMS = [
    # Baseball / MLB
    "mlb", "major league", "baseball", "spring training", "cactus league", "grapefruit league",
    "opening day", "postseason", "playoffs", "world series", "nl west", "national league",
    # Transactions / roster
    "trade", "traded", "acquired", "deal", "deadline",
    "dfa", "designated for assignment", "waivers", "claimed",
    "optioned", "option", "call-up", "called up", "sent down",
    "roster", "40-man", "40 man", "injured list", "il", "rehab assignment",
    # Roles / team building
    "pitcher", "starter", "rotation", "bullpen", "reliever", "closer",
    "catcher", "shortstop", "second base", "third base", "outfield", "first base", "dh",
    # Stats-y words
    "inning", "innings", "era", "fip", "whip", "strikeout", "strikeouts", "walks", "ks",
    "home run", "homer", "batting", "slugging", "ops", "wrc+", "war", "xwoba",
    # Prospects / org
    "prospect", "prospects", "farm system", "scouting", "draft", "international signing",
    "player development", "minor league", "triple-a", "double-a",
    # Coaching/FO terms
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
    # Transactions
    "trade", "traded", "acquire", "acquired", "deal", "waiver", "waivers",
    "claimed", "dfa", "designated for assignment", "optioned", "call-up", "called up",
    "sign", "signed", "signing", "extension",
    # Health
    "injury", "injured", "il", "injured list", "surgery", "rehab",
    # Prospects / callups
    "prospect", "prospects", "promotion", "promoted",
    # Team building
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
    # meta
    is_primary: bool
    score: int = 0
    raw_summary: str = ""


# -----------------------------
# Helpers
# -----------------------------
RE_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
RE_SPACE = re.compile(r"\s+")


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
    return requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": UA},
        allow_redirects=True,
    )


def title_hash(title: str) -> str:
    t = re.sub(r"[^a-z0-9 ]+", "", (title or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def extract_publication_from_title(raw_title: str) -> Tuple[str, str]:
    """
    Many Google News RSS items are like: "Headline - Publication"
    Return (headline, publication)
    """
    t = (raw_title or "").strip()
    parts = [p.strip() for p in t.split(" - ") if p.strip()]
    if len(parts) >= 2 and len(parts[-1]) <= 60:
        return " - ".join(parts[:-1]).strip(), parts[-1].strip()
    return t, ""


def parse_entry_datetime(entry: Any) -> Optional[datetime]:
    # feedparser structs
    for key in ("published_parsed", "updated_parsed"):
        st = getattr(entry, key, None)
        if st:
            try:
                return datetime.fromtimestamp(time.mktime(st), tz=timezone.utc)
            except Exception:
                pass

    # strings
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


def google_news_rss_url(query: str) -> str:
    q = quote(query)
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def is_google_host(url: str) -> bool:
    host = domain_of(url)
    return ("news.google.com" in host) or ("google.com" in host)


def pick_best_url(entry: Any) -> str:
    """
    Prefer non-Google link in entry.links, then non-Google href in summary, else entry.link.
    """
    for l in entry.get("links", []) or []:
        href = l.get("href")
        if href and not is_google_host(href):
            return href

    summary = entry.get("summary", "") or entry.get("description", "") or ""
    hrefs = re.findall(r'href="(https?://[^"]+)"', summary)
    for href in hrefs:
        if href and not is_google_host(href):
            return href

    return entry.get("link", "") or ""


def resolve_url(url: str) -> str:
    """
    Follow redirects once to reach the canonical URL when possible.
    """
    u = (url or "").strip()
    if not u:
        return u
    try:
        r = safe_get(u, timeout=20)
        if r.url:
            return r.url
    except Exception:
        pass
    return u


# -----------------------------
# State
# -----------------------------
def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {"posted": {}, "roster_cache": {}, "staff_cache": {}, "daily_other": {}}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)
    state.setdefault("posted", {})
    state.setdefault("roster_cache", {})
    state.setdefault("staff_cache", {})
    state.setdefault("daily_other", {})
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
    state[key] = {
        "fetched_at": utcnow().isoformat(),
        "names": sorted(names),
    }


# -----------------------------
# MLB Stats API: roster + (best effort) coaches
# -----------------------------
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
    """
    Best-effort. MLB Stats API sometimes exposes a coaches endpoint; shape varies.
    If it fails, return empty set.
    """
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
# Name matching (full name + tight last name)
# -----------------------------
def tokenize_words(text: str) -> Set[str]:
    return set(w.lower() for w in RE_WORD.findall(text or ""))


def last_name_token(full_name: str) -> Optional[str]:
    parts = [p for p in RE_WORD.findall(full_name or "") if p]
    if not parts:
        return None
    return parts[-1].lower()


def build_name_matchers(names: Set[str]) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    full_names: lowercased full names
    last_name_map: last name -> set(full names)
    """
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
    """
    Tight last-name rule:
    - last name must appear as a token
    - AND must include at least one anchor that suggests Giants/baseball context
    """
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
    for ln in last_name_map.keys():
        if ln in words:
            return True
    return False


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


def is_allowed_item(title: str, summary: str, publication: str,
                    full_names: Set[str], last_map: Dict[str, Set[str]]) -> bool:
    blob = f"{title}\n{summary}"

    if has_negative(blob):
        return False

    if not has_baseball_context(blob, publication):
        return False

    # Strong team mention qualifies
    if mentions_team_strong(blob):
        return True

    # Otherwise must match a known relevant name (roster/staff/front office/key people)
    if full_names and (mentions_full_name(blob, full_names) or mentions_last_name_tight(blob, last_map)):
        return True

    return False


def importance_score(title: str, summary: str, publication: str,
                     full_names: Set[str], key_people: Set[str],
                     published: datetime, echo_count: int) -> int:
    blob = f"{title} {summary}".lower()
    score = 0

    # Big: front office / coaching / VIP names
    for name in key_people:
        if name.lower() in blob:
            score += 10

    # Player/staff full-name mention
    if any(n in blob for n in full_names):
        score += 6

    # Strong team mention
    if ("san francisco giants" in blob) or ("sf giants" in blob):
        score += 4

    # Important baseball keywords
    for kw in IMPORTANT_KEYWORDS:
        if kw in blob:
            score += 2

    # Echo bonus (appears multiple times across feeds)
    if echo_count >= 2:
        score += min(8, 3 * (echo_count - 1))

    # Recency bonus
    hours_old = max(0.0, (utcnow() - published).total_seconds() / 3600.0)
    if hours_old <= 2:
        score += 3
    elif hours_old <= 6:
        score += 2
    elif hours_old <= 12:
        score += 1

    # Slight bonus for publication that’s already baseball-y
    pl = (publication or "").lower()
    if any(h in pl for h in BASEBALL_SOURCE_HINTS):
        score += 1

    return score


# -----------------------------
# Feeds
# -----------------------------
def fetch_feed_items(feed_url: str, source_label: str,
                     cutoff: datetime,
                     full_names: Set[str], last_map: Dict[str, Set[str]]) -> List[Item]:
    """
    Fetch via requests (so UA is consistent) then parse with feedparser.
    Apply tight relevance, and return candidate Items.
    """
    r = safe_get(feed_url, timeout=30)
    r.raise_for_status()

    fp = feedparser.parse(r.text)
    items: List[Item] = []

    for e in fp.entries:
        dt = parse_entry_datetime(e)
        if not dt or dt < cutoff:
            continue

        raw_title = (e.get("title") or "").strip()
        if not raw_title:
            continue

        headline, pub_from_title = extract_publication_from_title(raw_title)
        publication = pub_from_title or source_label

        raw_url = pick_best_url(e)
        url = resolve_url(raw_url)
        if not url:
            continue

        d = domain_of(url)
        if not d or d in AGGREGATOR_BLOCKLIST:
            continue

        summary = e.get("summary", "") or e.get("description", "") or ""

        if not is_allowed_item(headline, summary, publication, full_names, last_map):
            continue

        is_primary = any(d == pd or d.endswith("." + pd) for pd in PRIMARY_DOMAINS)

        items.append(Item(
            title=headline,
            url=url,
            publication=publication,
            published=dt,
            domain=d,
            is_primary=is_primary,
            raw_summary=summary,
        ))

    return items


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


def bsky_post(access_jwt: str, did: str, text: str, url: str, title: str, publication: str) -> None:
    record = {
        "$type": "app.bsky.feed.post",
        "text": text,
        "createdAt": utcnow().isoformat().replace("+00:00", "Z"),
        "embed": {
            "$type": "app.bsky.embed.external",
            "external": {
                "uri": url,
                "title": (title or "")[:300],
                "description": (publication or "")[:300],
            },
        },
    }

    r = requests.post(
        f"{BSKY_PDS}/xrpc/com.atproto.repo.createRecord",
        json={"repo": did, "collection": "app.bsky.feed.post", "record": record},
        timeout=20,
        headers={"Authorization": f"Bearer {access_jwt}", "User-Agent": UA},
    )
    if r.status_code >= 400:
        print(f"[error] createRecord {r.status_code}: {r.text[:2000]}")
    r.raise_for_status()


def format_post_text(title: str, publication: str) -> str:
    # Keep it clean: no URL in text (embed card handles link)
    pub = (publication or "").strip()
    t = (title or "").strip()
    text = f"{t}\n{pub}".strip()

    if len(text) <= 300:
        return text

    room_for_title = max(20, 300 - (len(pub) + 1))
    if len(t) > room_for_title:
        t = t[: room_for_title - 1].rstrip() + "…"
    return f"{t}\n{pub}".strip()


# -----------------------------
# Main
# -----------------------------
def main():
    state = load_state()
    prune_state(state)
    daily_other = get_daily_other_counter(state)

    cutoff = utcnow() - timedelta(hours=HOURS_BACK)

    # --- Load / refresh roster + staff caches
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
            # Cache even if empty to avoid hammering
            save_cached_names(state, "staff_cache", staff_names)
            print(f"[info] staff cached: {len(staff_names)} names")
        except Exception as e:
            print(f"[warn] staff fetch failed: {e}")
            staff_names = set()

    # --- Build the name universe: roster + staff + key people
    all_names: Set[str] = set()
    all_names.update(roster_names)
    all_names.update(staff_names)
    all_names.update(KEY_PEOPLE)

    full_names, last_map = build_name_matchers(all_names)

    # ---------
    # FEEDS
    # ---------
    feeds: List[Tuple[str, str]] = []

    # Direct RSS
    feeds.append(("SF Standard", "https://sfstandard.com/sports/feed"))
    feeds.append(("SFGate", "https://www.sfgate.com/sports/feed/san-francisco-giants-rss-feed-428.php"))

    # Optional: KNBR podcast feed (often not what you want for “articles”)
    # feeds.append(("KNBR (Podcast)", "https://www.omnycontent.com/d/playlist/a7b0bd27-d748-4fbe-ab3b-a6fa0049bcf6/e36ffdc5-c7fa-4985-b94c-a8bd01670702/808a103c-058f-4c12-a1ae-a8bd01675a7c/podcast.rss"))

    # Your chosen domains via Google News RSS (baseball-aware)
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

    # “Best of the rest” broad search (outside your domains, capped to OTHER_DAILY_CAP/day)
    feeds.append(("Google News: Broad", google_news_rss_url(
        '(("San Francisco Giants" OR "SF Giants") AND (MLB OR baseball))'
    )))

    # ---------
    # Collect items (already time-filtered in fetch)
    # ---------
    all_items: List[Item] = []
    for source_label, feed_url in feeds:
        try:
            items = fetch_feed_items(feed_url, source_label, cutoff, full_names, last_map)
            print(f"[info] feed: {source_label} -> {len(items)} eligible")
            all_items.extend(items)
        except Exception as ex:
            print(f"[warn] feed failed: {source_label}: {ex}")

    if not all_items:
        print("No eligible items found.")
        save_state(state)
        return

    # Echo counts (title similarity proxy)
    echo_map: Dict[str, int] = {}
    for it in all_items:
        h = title_hash(it.title)
        echo_map[h] = echo_map.get(h, 0) + 1

    # Score everything (score mainly used for selecting “other sources”)
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

    # Sort newest first, but we'll later sort “other” by score
    all_items.sort(key=lambda x: x.published, reverse=True)

    posted = state.get("posted", {})
    seen_urls: Set[str] = set()

    # Split into primary vs other
    primary_candidates: List[Item] = []
    other_candidates: List[Item] = []
    for it in all_items:
        # Hard skip if already posted
        if it.url in posted:
            continue
        if it.url in seen_urls:
            continue
        seen_urls.add(it.url)

        # Determine "primary-ness" by final URL domain
        is_primary = any(it.domain == d or it.domain.endswith("." + d) for d in PRIMARY_DOMAINS)
        it.is_primary = is_primary

        if is_primary:
            primary_candidates.append(it)
        else:
            other_candidates.append(it)

    # Rank “other” by score (desc), then recency
    other_candidates.sort(key=lambda x: (x.score, x.published), reverse=True)

    # Selection with per-publication cap
    per_pub_count: Dict[str, int] = {}
    to_post: List[Item] = []

    def pub_key(pub: str) -> str:
        return norm_text(pub)

    # 1) Fill from primary first
    for it in primary_candidates:
        if len(to_post) >= MAX_POSTS_PER_RUN:
            break
        pk = pub_key(it.publication)
        per_pub_count.setdefault(pk, 0)
        if per_pub_count[pk] >= PER_SOURCE_CAP:
            continue
        to_post.append(it)
        per_pub_count[pk] += 1

    # 2) Then fill from “other sources” up to daily cap
    # Only if we still have room
    for it in other_candidates:
        if len(to_post) >= MAX_POSTS_PER_RUN:
            break
        if daily_other.get("count", 0) >= OTHER_DAILY_CAP:
            break

        pk = pub_key(it.publication)
        per_pub_count.setdefault(pk, 0)
        if per_pub_count[pk] >= PER_SOURCE_CAP:
            continue

        to_post.append(it)
        per_pub_count[pk] += 1

        # Increment daily other counter only when we actually POST successfully (done later)
        # For now, just tentatively include.

    if not to_post:
        print("No new items to post.")
        save_state(state)
        return

    # ---------
    # Post
    # ---------
    sess = bsky_create_session()
    access_jwt = sess["accessJwt"]
    did = sess["did"]

    posted_any = 0
    for it in to_post:
        text = format_post_text(it.title, it.publication)
        try:
            print(f"[post] ({it.score}) {it.publication}: {it.title} -> {it.url}")
            bsky_post(access_jwt, did, text, it.url, it.title, it.publication)
            posted[it.url] = utcnow().isoformat()
            posted_any += 1

            # If this was an “other” source (not in PRIMARY_DOMAINS), count it against the daily cap
            if not it.is_primary:
                daily_other["count"] = int(daily_other.get("count", 0)) + 1

        except Exception as e:
            print(f"[warn] post failed: {e}")

    state["posted"] = posted
    state["daily_other"] = daily_other
    save_state(state)

    if posted_any == 0:
        raise SystemExit("All posts failed.")
    print(f"Posted {posted_any} items.")


if __name__ == "__main__":
    main()
