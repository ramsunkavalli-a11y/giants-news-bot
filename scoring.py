from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from config import PRIORITY_AUTHORS, SOURCE_TRUST
from filters import giants_relevance_signals, normalize_author, parse_dt_or_none, suspicious_candidate
from models import Candidate


def score_candidate(candidate: Candidate) -> Candidate:
    components: dict[str, int] = {}

    # Provisional score should always exist.
    base = 1
    components["base"] = base

    trust = SOURCE_TRUST.get(candidate.source, 0)
    components["source_trust"] = trust

    signals = giants_relevance_signals(candidate.title, candidate.summary, candidate.categories, candidate.url)
    if signals["strong_giants"]:
        components["strong_giants"] = 4
    elif signals["giants_baseball"]:
        components["giants_baseball"] = 3
    else:
        components["weak_relevance"] = -3

    norm_author = normalize_author(candidate.author)
    if norm_author and norm_author in PRIORITY_AUTHORS:
        components["priority_author"] = 2

    dt = parse_dt_or_none(candidate.published_ts)
    if dt and dt >= datetime.now(timezone.utc) - timedelta(hours=6):
        components["very_recent"] = 1

    path = urlparse(candidate.url).path.lower()
    if any(seg in path for seg in ["/article/", "/giants/news/"]) or path.endswith(".html"):
        components["article_like_path"] = 1
    else:
        components["weak_path"] = -1

    if signals["nfl_signal"]:
        components["nfl_penalty"] = -6

    if suspicious_candidate(candidate.url, candidate.title, candidate.summary):
        components["low_confidence"] = -2

    # Enriched score add-ons.
    if candidate.discovered_via == "rss":
        components["rss_preference"] = 2
    if candidate.article_meta_confirmed:
        components["article_meta_confirmed"] = 2
    else:
        components["meta_unconfirmed"] = -1

    candidate.score_components = components
    candidate.score = sum(components.values())
    return candidate
