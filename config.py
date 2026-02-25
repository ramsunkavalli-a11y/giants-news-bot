from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SourceConfig:
    name: str
    discovery_mode: str  # rss_only | google_news | non_rss
    trust_tier: int = 2
    rss_url: str = ""
    google_query: str = ""
    listing_url: str = ""
    max_candidates: int = 40


@dataclass
class Settings:
    user_agent: str = os.getenv(
        "USER_AGENT",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 GiantsNewsBot/2026",
    )
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "15"))
    dry_run: bool = os.getenv("DRY_RUN", "0").lower() in {"1", "true", "yes"}
    state_file: str = os.getenv("STATE_FILE", "state.json")
    diagnostics_enabled: bool = os.getenv("DIAGNOSTICS_ENABLED", "0").lower() in {"1", "true", "yes"}
    diagnostics_file: str = os.getenv("DIAGNOSTICS_FILE", "diagnostics.json")
    max_posts_per_run: int = int(os.getenv("MAX_POSTS_PER_RUN", "10"))
    max_per_source_per_run: int = int(os.getenv("MAX_PER_SOURCE_PER_RUN", "3"))
    hours_back: int = int(os.getenv("HOURS_BACK", "24"))
    keep_posted_days: int = int(os.getenv("KEEP_POSTED_DAYS", "21"))
    meta_cache_days: int = int(os.getenv("META_CACHE_DAYS", os.getenv("KEEP_POSTED_DAYS", "21")))
    max_non_rss_urls_per_source: int = int(os.getenv("MAX_NON_RSS_URLS_PER_SOURCE", "60"))
    max_rss_entries_per_feed: int = int(os.getenv("MAX_RSS_ENTRIES_PER_FEED", "40"))
    score_threshold: int = int(os.getenv("SCORE_THRESHOLD", "2"))
    bsky_pds: str = os.getenv("BSKY_PDS", "https://bsky.social")
    bsky_identifier: str = os.getenv("BSKY_IDENTIFIER", "")
    bsky_app_password: str = os.getenv("BSKY_APP_PASSWORD", "")
    tracking_query_keys: set[str] = field(
        default_factory=lambda: {
            "fbclid",
            "gclid",
            "ref",
            "refsrc",
            "mc_cid",
            "mc_eid",
            "igshid",
            "source",
        }
    )


def load_priority_authors() -> set[str]:
    defaults = {
        "justice delos santos",
        "alex pavlovic",
        "john shea",
        "shayna rubin",
        "susan slusser",
        "janie mccauley",
        "andrew baggarly",
        "grant brisbee",
    }
    raw = os.getenv("AUTHOR_PRIORITY_JSON", "").strip()
    if not raw:
        return defaults
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return defaults
        high = payload.get("high", [])
        if not isinstance(high, list):
            return defaults
        parsed = {str(v).strip().lower() for v in high if str(v).strip()}
        return parsed or defaults
    except Exception:
        return defaults


PRIORITY_AUTHORS = load_priority_authors()


RSS_ONLY_SOURCES: List[SourceConfig] = [
    SourceConfig(
        name="SF Standard",
        discovery_mode="rss_only",
        rss_url="https://sfstandard.com/category/sports/feed/",
        trust_tier=3,
    ),
    SourceConfig(
        name="SFGate Giants",
        discovery_mode="rss_only",
        rss_url="https://www.sfgate.com/giants/feed/Giants-447.php",
        trust_tier=3,
    ),
    SourceConfig(
        name="NYTimes Baseball",
        discovery_mode="rss_only",
        rss_url="https://rss.nytimes.com/services/xml/rss/nyt/Baseball.xml",
        trust_tier=2,
    ),
]

GOOGLE_NEWS_SOURCES: List[SourceConfig] = [
    SourceConfig(name="NBC Sports Bay Area", discovery_mode="google_news", google_query='site:nbcsportsbayarea.com "San Francisco Giants"', trust_tier=2),
    SourceConfig(name="SF Chronicle Giants", discovery_mode="google_news", google_query='site:sfchronicle.com "San Francisco Giants"', trust_tier=2),
    SourceConfig(name="Mercury News Giants", discovery_mode="google_news", google_query='site:mercurynews.com "San Francisco Giants"', trust_tier=2),
    SourceConfig(name="AP Giants", discovery_mode="google_news", google_query='site:apnews.com "San Francisco Giants"', trust_tier=2),
    SourceConfig(name="MLB Giants", discovery_mode="google_news", google_query='site:mlb.com/giants/news "San Francisco Giants"', trust_tier=3),
    SourceConfig(name="Fangraphs Giants", discovery_mode="google_news", google_query='site:blogs.fangraphs.com giants', trust_tier=2),
    SourceConfig(name="Baseball America Giants", discovery_mode="google_news", google_query='site:baseballamerica.com "San Francisco Giants"', trust_tier=2),
    SourceConfig(name="KNBR Giants", discovery_mode="google_news", google_query='site:knbr.com giants', trust_tier=2),
]

NON_RSS_SOURCES: List[SourceConfig] = [
    SourceConfig(name="NBC Sports Bay Area", discovery_mode="non_rss", listing_url="https://www.nbcsportsbayarea.com/mlb/san-francisco-giants/", trust_tier=1),
    SourceConfig(name="SF Chronicle Giants", discovery_mode="non_rss", listing_url="https://www.sfchronicle.com/sports/giants/", trust_tier=1),
    SourceConfig(name="Mercury News Giants", discovery_mode="non_rss", listing_url="https://www.mercurynews.com/tag/san-francisco-giants/", trust_tier=1),
    SourceConfig(name="AP Giants hub", discovery_mode="non_rss", listing_url="https://apnews.com/hub/san-francisco-giants", trust_tier=1),
    SourceConfig(name="MLB Giants News", discovery_mode="non_rss", listing_url="https://www.mlb.com/giants/news", trust_tier=1),
    SourceConfig(name="Fangraphs Giants", discovery_mode="non_rss", listing_url="https://blogs.fangraphs.com/category/giants/", trust_tier=1),
    SourceConfig(name="Baseball America Giants", discovery_mode="non_rss", listing_url="https://www.baseballamerica.com/teams/2003/san-francisco-giants/", trust_tier=1),
    SourceConfig(name="KNBR Giants", discovery_mode="non_rss", listing_url="https://www.knbr.com/category/giants/", trust_tier=1),
]

ALL_SOURCES = RSS_ONLY_SOURCES + GOOGLE_NEWS_SOURCES + NON_RSS_SOURCES
SOURCE_TRUST: Dict[str, int] = {s.name: s.trust_tier for s in ALL_SOURCES}
RSS_ONLY_NAMES = {s.name for s in RSS_ONLY_SOURCES}

AGGREGATOR_BLOCKLIST = {
    "news.google.com",
    "google.com",
    "feedly.com",
    "feedspot.com",
    "newsbreak.com",
}

SUSPICIOUS_HOST_TERMS = {"yahoo", "msn", "flipboard"}
