import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from urllib.parse import quote

import feedparser
import requests
from dateutil import parser as dtparser


# -----------------------------
# Config
# -----------------------------
HOURS_BACK = int(os.getenv("HOURS_BACK", "8"))
MAX_POSTS_PER_RUN = int(os.getenv("MAX_POSTS_PER_RUN", "5"))
STATE_FILE = os.getenv("STATE_FILE", "state.json")

BSKY_IDENTIFIER = os.environ["BSKY_IDENTIFIER"]     # handle or email
BSKY_APP_PASSWORD = os.environ["BSKY_APP_PASSWORD"]

# If you use a custom PDS, set it here; otherwise bsky.social is fine.
BSKY_PDS = os.getenv("BSKY_PDS", "https://bsky.social")

UA = "GiantsNewsBot/1.0 (+github-actions)"


@dataclass
class Item:
    title: str
    url: str
    source: str
    published: datetime


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {"posted": {}}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def prune_state(state: Dict[str, Any], keep_days: int = 14) -> None:
    cutoff = utcnow() - timedelta(days=keep_days)
    posted = state.get("posted", {})
    to_del = []
    for url, ts in posted.items():
        try:
            dt = dtparser.isoparse(ts)
            if dt < cutoff:
                to_del.append(url)
        except Exception:
            to_del.append(url)
    for k in to_del:
        posted.pop(k, None)
    state["posted"] = posted


def safe_get(url: str, timeout: int = 20) -> requests.Response:
    return requests.get(url, timeout=timeout, headers={"User-Agent": UA}, allow_redirects=True)


def extract_first_href(html: str) -> Optional[str]:
    # Google News RSS often includes the real source link in the description/summary.
    m = re.search(r'href="(https?://[^"]+)"', html)
    return m.group(1) if m else None


def resolve_url(maybe_wrapped_url: str, entry_summary: Optional[str] = None) -> str:
    # Best effort:
    # 1) If summary has a direct href, prefer it.
    if entry_summary:
        href = extract_first_href(entry_summary)
        if href:
            return href

    # 2) Try following redirects
    try:
        r = safe_get(maybe_wrapped_url, timeout=20)
        if r.url:
            return r.url
    except Exception:
        pass

    return maybe_wrapped_url


def parse_entry_datetime(entry: Any) -> Optional[datetime]:
    # Try feedparser's parsed structs first
    for key in ("published_parsed", "updated_parsed"):
        if hasattr(entry, key) and getattr(entry, key):
            st = getattr(entry, key)
            try:
                return datetime.fromtimestamp(time.mktime(st), tz=timezone.utc)
            except Exception:
                pass

    # Fallback to strings
    for key in ("published", "updated"):
        if key in entry and entry.get(key):
            try:
                dt = dtparser.parse(entry.get(key))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass
    return None


def google_news_rss_url(query: str) -> str:
    # Common pattern for Google News RSS search feeds.
    # Add locale parameters for stability.
    q = quote(query)
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def fetch_feed_items(feed_url: str, source_name: str) -> List[Item]:
    fp = feedparser.parse(feed_url)
    items: List[Item] = []

    for e in fp.entries:
        dt = parse_entry_datetime(e)
        if not dt:
            continue

        url = resolve_url(e.get("link", ""), e.get("summary"))
        title = (e.get("title") or "").strip()
        if not title or not url:
            continue

        items.append(Item(title=title, url=url, source=source_name, published=dt))

    return items


def bsky_create_session() -> Dict[str, Any]:
    r = requests.post(
        f"{BSKY_PDS}/xrpc/com.atproto.server.createSession",
        json={"identifier": BSKY_IDENTIFIER, "password": BSKY_APP_PASSWORD},
        timeout=20,
        headers={"User-Agent": UA},
    )
    r.raise_for_status()
    return r.json()


def bsky_post(access_jwt: str, did: str, text: str, url: str, title: str, source: str) -> None:
    # External embed card (nice in Bluesky clients)
    record = {
        "$type": "app.bsky.feed.post",
        "text": text,
        "createdAt": utcnow().isoformat().replace("+00:00", "Z"),
        "embed": {
            "$type": "app.bsky.embed.external",
            "external": {
                "uri": url,
                "title": title[:300],
                "description": source[:300],
            },
        },
    }

    r = requests.post(
        f"{BSKY_PDS}/xrpc/com.atproto.repo.createRecord",
        json={"repo": did, "collection": "app.bsky.feed.post", "record": record},
        timeout=20,
        headers={"Authorization": f"Bearer {access_jwt}", "User-Agent": UA},
    )
    r.raise_for_status()


def format_post_text(title: str, source: str) -> str:
    # Keep text short; the link is in the embed card.
    text = f"{title}\n{source}"
    if len(text) <= 300:
        return text

    # Trim title if needed
    # Leave room for newline + source
    room_for_title = max(20, 300 - (len(source) + 1))
    t = title
    if len(t) > room_for_title:
        t = t[: room_for_title - 1].rstrip() + "…"
    return f"{t}\n{source}"


def main():
    state = load_state()
    state.setdefault("posted", {})
    prune_state(state)

    cutoff = utcnow() - timedelta(hours=HOURS_BACK)

    # ---------
    # FEEDS
    # ---------
    feeds = []

    # Direct RSS
    feeds.append(("SF Standard (Sports)", "https://sfstandard.com/sports/feed"))
    feeds.append(("SFGate (Giants)", "https://www.sfgate.com/sports/feed/san-francisco-giants-rss-feed-428.php"))
    # Optional: KNBR podcast feed (clips / audio)
    feeds.append(("KNBR (Podcast)", "https://www.omnycontent.com/d/playlist/a7b0bd27-d748-4fbe-ab3b-a6fa0049bcf6/e36ffdc5-c7fa-4985-b94c-a8bd01670702/808a103c-058f-4c12-a1ae-a8bd01675a7c/podcast.rss"))

    # Google News RSS per-domain searches (fallbacks / “no RSS” publishers)
    domain_queries = [
        ("SF Chronicle", '("San Francisco Giants" OR "SF Giants") site:sfchronicle.com'),
        ("Mercury News", '("San Francisco Giants" OR "SF Giants") site:mercurynews.com'),
        ("NBC Sports Bay Area", '("San Francisco Giants" OR "SF Giants") site:nbcsportsbayarea.com'),
        ("The Athletic", '("San Francisco Giants" OR "SF Giants") site:theathletic.com'),
        ("Associated Press", '("San Francisco Giants" OR "SF Giants") site:apnews.com'),
        ("SFGiants.com", '("San Francisco Giants" OR "SF Giants") site:mlb.com/giants OR site:sfgiants.com'),
    ]
    for name, q in domain_queries:
        feeds.append((f"Google News: {name}", google_news_rss_url(q)))

    # Collect items
    all_items: List[Item] = []
    for source_name, feed_url in feeds:
        try:
            items = fetch_feed_items(feed_url, source_name)
            all_items.extend(items)
        except Exception as ex:
            print(f"[warn] feed failed: {source_name}: {ex}")

    # Filter by time + dedupe
    fresh = [it for it in all_items if it.published >= cutoff]
    # Sort newest first
    fresh.sort(key=lambda x: x.published, reverse=True)

    posted = state["posted"]
    to_post: List[Item] = []
    seen_urls = set()

    for it in fresh:
        u = it.url.strip()
        if not u or u in posted or u in seen_urls:
            continue
        seen_urls.add(u)
        to_post.append(it)
        if len(to_post) >= MAX_POSTS_PER_RUN:
            break

    if not to_post:
        print("No new items to post.")
        save_state(state)
        return

    sess = bsky_create_session()
    access_jwt = sess["accessJwt"]
    did = sess["did"]

    for it in to_post:
text = format_post_text(it.title, it.source)
bsky_post(access_jwt, did, text, it.url, it.title, it.source)


    state["posted"] = posted
    save_state(state)
    print(f"Posted {len(to_post)} items.")


if __name__ == "__main__":
    main()
