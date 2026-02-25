from __future__ import annotations

import html as html_lib
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from dateutil import parser as dtparser

from config import AGGREGATOR_BLOCKLIST, SUSPICIOUS_HOST_TERMS, Settings


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
    "mlb",
    "baseball",
    "pitcher",
    "inning",
    "oracle park",
    "nl west",
    "rotation",
    "bullpen",
    "hitter",
}
NFL_TERMS = {"new york giants", "quarterback", "touchdown", "nfl", "super bowl", "metlife"}


TAG_RE = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    if not text:
        return ""
    txt = TAG_RE.sub(" ", text)
    txt = html_lib.unescape(txt)
    return re.sub(r"\s+", " ", txt).strip()


def normalize_author(author: str) -> str:
    s = clean_text(author or "").lower().strip()
    if s.startswith("by "):
        s = s[3:].strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def canonicalize_url(url: Any, settings: Settings) -> str:
    if not isinstance(url, str) or not url:
        return ""
    parsed = urlparse(url.strip())
    filtered = []
    for k, v in parse_qsl(parsed.query, keep_blank_values=True):
        kl = k.lower()
        if kl in settings.tracking_query_keys or kl.startswith("utm_") or kl.startswith("mc_"):
            continue
        filtered.append((k, v))
    cleaned = parsed._replace(netloc=parsed.netloc.lower(), query=urlencode(filtered, doseq=True), fragment="")
    return urlunparse(cleaned)


def parse_dt_or_none(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = dtparser.isoparse(value)
    except Exception:
        try:
            dt = dtparser.parse(value)
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def is_recent_enough(published_at: str, hours_back: int) -> bool:
    dt = parse_dt_or_none(published_at)
    if not dt:
        return True
    return dt >= datetime.now(timezone.utc) - timedelta(hours=hours_back)


def is_bad_domain(domain: str) -> bool:
    d = (domain or "").lower()
    return d in AGGREGATOR_BLOCKLIST or d.endswith(".google.com")


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
    segs = [s for s in path.split("/") if s]
    if len(segs) <= 1:
        return False
    if any(seg in STORY_REJECT_SEGMENTS for seg in segs):
        return False
    if segs[-1] in STORY_REJECT_SLUGS:
        return False
    if re.search(r"/20\d{2}/\d{1,2}/\d{1,2}/", "/" + path + "/"):
        return True
    if segs[-1].endswith(".html") or "/article/" in "/" + path + "/":
        return True
    return "-" in segs[-1] and len(segs[-1]) >= 20


def giants_relevance_signals(title: str, summary: str, categories: Iterable[str], url: str) -> dict[str, bool]:
    text = " ".join([clean_text(title), clean_text(summary), " ".join(categories or []), url or ""]).lower()
    strong = "san francisco giants" in text or "sf giants" in text
    mention = "giants" in text
    baseball_context = any(t in text for t in BASEBALL_TERMS)
    nfl_signal = any(t in text for t in NFL_TERMS)
    return {
        "strong_giants": strong,
        "giants_baseball": mention and baseball_context,
        "nfl_signal": nfl_signal,
    }


def is_relevant_giants(title: str, summary: str, categories: Iterable[str], url: str) -> bool:
    signals = giants_relevance_signals(title, summary, categories, url)
    if signals["nfl_signal"]:
        return False
    return signals["strong_giants"] or signals["giants_baseball"]


def suspicious_candidate(url: str, title: str, summary: str) -> bool:
    parsed = urlparse(url)
    text = f"{title} {summary}".lower()
    return any(term in parsed.netloc for term in SUSPICIOUS_HOST_TERMS) or len(clean_text(text)) < 30
