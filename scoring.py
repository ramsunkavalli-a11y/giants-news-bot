from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from config import PRIORITY_AUTHORS, SOURCE_TRUST
from filters import giants_relevance_signals, normalize_author, parse_dt_or_none, suspicious_candidate
from models import Candidate


def score_candidate(candidate: Candidate) -> Candidate:
    components: dict[str, int] = {}

    if SOURCE_TRUST.get(candidate.source, 0) >= 2:
        components["trusted_source"] = 3

    signals = giants_relevance_signals(candidate.title, candidate.summary, candidate.categories, candidate.url)
    if signals["strong_giants"]:
        components["strong_giants"] = 3
    elif signals["giants_baseball"]:
        components["giants_baseball"] = 2

    norm_author = normalize_author(candidate.author)
    if norm_author:
        if norm_author in PRIORITY_AUTHORS:
            components["priority_author"] = 2
    else:
        components["author_missing"] = 0

    dt = parse_dt_or_none(candidate.published_ts)
    if dt and dt >= datetime.now(timezone.utc) - timedelta(hours=6):
        components["very_recent"] = 1

    if signals["nfl_signal"]:
        components["nfl_penalty"] = -4

    if suspicious_candidate(candidate.url, candidate.title, candidate.summary):
        components["low_confidence"] = -2

    parsed = urlparse(candidate.url)
    if any(seg in parsed.path.lower() for seg in ["/tag/", "/topic/", "/video/"]):
        components["non_story_penalty"] = -2

    if candidate.discovered_via == "rss":
        components["rss_preference"] = 1

    candidate.score_components = components
    candidate.score = sum(components.values())
    return candidate
