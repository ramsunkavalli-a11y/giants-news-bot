from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from dateutil import parser as dtparser

from v2_story import candidate_preference_key

PACIFIC = ZoneInfo("America/Los_Angeles")
BASEBALL_DAY_SHIFT_HOURS = 12
DEFAULT_GAME_HOURS_BACK = 30

# These aliases are only used to label/group a game-coverage thread. Failure to
# identify an opponent falls back to a date-only key; it never blocks a story.
TEAM_ALIASES = {
    "diamondbacks": ("diamondbacks", "d-backs", "arizona"),
    "athletics": ("athletics", "a's"),
    "braves": ("braves", "atlanta"),
    "orioles": ("orioles", "baltimore"),
    "red-sox": ("red sox", "boston"),
    "cubs": ("cubs", "chicago cubs"),
    "white-sox": ("white sox", "chicago white sox"),
    "reds": ("reds", "cincinnati"),
    "guardians": ("guardians", "cleveland"),
    "rockies": ("rockies", "colorado"),
    "tigers": ("tigers", "detroit"),
    "astros": ("astros", "houston"),
    "royals": ("royals", "kansas city"),
    "angels": ("angels", "la angels", "los angeles angels"),
    "dodgers": ("dodgers", "la dodgers", "los angeles dodgers"),
    "marlins": ("marlins", "miami"),
    "brewers": ("brewers", "milwaukee"),
    "twins": ("twins", "minnesota"),
    "mets": ("mets", "ny mets", "new york mets"),
    "yankees": ("yankees", "ny yankees", "new york yankees"),
    "phillies": ("phillies", "philadelphia"),
    "pirates": ("pirates", "pittsburgh"),
    "padres": ("padres", "san diego"),
    "mariners": ("mariners", "seattle"),
    "cardinals": ("cardinals", "st. louis", "st louis"),
    "rays": ("rays", "tampa bay"),
    "rangers": ("rangers", "texas rangers"),
    "blue-jays": ("blue jays", "toronto"),
    "nationals": ("nationals", "washington nationals"),
}

GAME_TITLE_PATTERNS = (
    "what we learned",
    "observations",
    "takeaways:",
    "takeaways ",
    " in win",
    " in loss",
    " win over ",
    " loss to ",
    " falls to ",
    " fall to ",
    " doom ",
    "earns win",
    "earned win",
    "quality start",
    "solid start",
    "sharp start",
    "sterling start",
    "strong start",
)
RESULT_VERBS = re.compile(
    r"\b(?:lead|leads|lift|lifts|power|powers|propel|propels|beat|beats|edge|edges|"
    r"defeat|defeats|top|tops|rout|routs)\b.*\b(?:over|past)\b",
    flags=re.I,
)
GIANTS_RESULT = re.compile(
    r"(?:\bgiants\b.*\b(?:win|loss|victory|defeat)\b|"
    r"\b(?:win|loss|victory|defeat)\b.*\bgiants\b)",
    flags=re.I,
)


def _parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = dtparser.parse(value, tzinfos={"UT": timezone.utc})
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_game_story(article: dict) -> bool:
    if article.get("content_type") == "game_story":
        return True
    if article.get("quality_reason") == "game_story_or_postgame_analysis":
        return True
    title = str(article.get("title", "") or "")
    summary = str(article.get("summary", "") or "")
    blob = f"{title} {summary}"
    lower = blob.lower()
    return (
        any(pattern in lower for pattern in GAME_TITLE_PATTERNS)
        or bool(RESULT_VERBS.search(blob))
        or bool(GIANTS_RESULT.search(blob))
    )


def baseball_day(article: dict) -> str:
    """Return the likely local game date, with a noon cutoff for next-day coverage."""
    dt = _parse_dt(article.get("published", ""))
    if dt is None:
        return ""
    shifted = dt.astimezone(PACIFIC) - timedelta(hours=BASEBALL_DAY_SHIFT_HOURS)
    return shifted.date().isoformat()


def _contains_alias(text: str, alias: str) -> bool:
    if alias == "a's":
        return bool(re.search(r"(?<![a-z])a['’]s(?![a-z])", text))
    return bool(re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", text))


def extract_opponent(article: dict) -> str:
    text = " ".join(
        str(article.get(field, "") or "")
        for field in ("title", "summary")
    ).lower().replace("’", "'")
    for canonical, aliases in TEAM_ALIASES.items():
        if any(_contains_alias(text, alias) for alias in aliases):
            return canonical
    return ""


def game_thread_key(game_day: str, opponent: str = "") -> str:
    return f"game:{game_day}:{opponent or 'unknown'}"


def group_game_articles(articles: list[dict]) -> list[dict]:
    """Group game coverage by baseball day/opponent without suppressing versions."""
    per_day: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for article in articles:
        day = baseball_day(article)
        if not day:
            continue
        opponent = extract_opponent(article)
        per_day[day][opponent].append(article)

    groups: list[dict] = []
    for day, opponent_groups in per_day.items():
        known = [opponent for opponent in opponent_groups if opponent]
        unknown = opponent_groups.pop("", [])
        if unknown and len(known) == 1:
            opponent_groups[known[0]].extend(unknown)
        elif unknown:
            opponent_groups[""] = unknown

        for opponent, members in opponent_groups.items():
            groups.append({
                "key": game_thread_key(day, opponent),
                "game_day": day,
                "opponent": opponent,
                "articles": sorted(members, key=candidate_preference_key, reverse=True),
            })

    def newest(group: dict) -> datetime:
        values = [_parse_dt(item.get("published", "")) for item in group["articles"]]
        return max((value for value in values if value is not None), default=datetime.min.replace(tzinfo=timezone.utc))

    groups.sort(key=newest, reverse=True)
    return groups


def select_game_threads(
    articles: list[dict],
    state: dict,
    *,
    hours_back: int = DEFAULT_GAME_HOURS_BACK,
    now: datetime | None = None,
) -> dict:
    """Select every fresh, non-low game story. Cross-publisher versions are kept."""
    # Local import avoids coupling the standalone selector back to this lane.
    from v2_selector import canonicalize_url, optional_page_metadata, parse_dt

    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours_back)
    raw_posted = state.get("posted_urls", {})
    if isinstance(raw_posted, dict):
        posted = {canonicalize_url(url) for url in raw_posted}
    elif isinstance(raw_posted, list):
        posted = {canonicalize_url(url) for url in raw_posted}
    else:
        posted = set()
    posted.discard("")

    reasons = Counter()
    diagnostics: list[dict] = []
    eligible: list[dict] = []
    timestamp_enrichments = 0

    for raw in articles:
        article = dict(raw)
        url = article.get("url", "")
        canonical = canonicalize_url(url)
        reason = "not_game_story"

        if not is_game_story(article):
            pass
        elif article.get("quality") == "low":
            reason = "quality_low"
        elif not canonical:
            reason = "missing_url"
        elif canonical in posted:
            reason = "already_posted"
        else:
            dt = parse_dt(article.get("published", ""))
            if dt is None and article.get("source") == "NBC Sports Bay Area" and timestamp_enrichments < 12:
                _, published = optional_page_metadata(url)
                timestamp_enrichments += 1
                if published:
                    article["published"] = published
                    dt = parse_dt(published)
            if dt is None:
                reason = "missing_published"
            elif dt < cutoff:
                reason = "stale_game_story"
            else:
                reason = "eligible_game_story"
                article["_published_dt"] = dt
                article["canonical_url"] = canonical
                eligible.append(article)

        reasons[reason] += 1
        if reason != "not_game_story":
            diagnostics.append({
                "source": article.get("source", ""),
                "title": article.get("title", ""),
                "url": url,
                "published": article.get("published", ""),
                "quality": article.get("quality", ""),
                "quality_reason": article.get("quality_reason", ""),
                "reason": reason,
            })

    threads = group_game_articles(eligible)

    def public(article: dict) -> dict:
        return {key: value for key, value in article.items() if not key.startswith("_")}

    public_threads = []
    for thread in threads:
        public_threads.append({
            "key": thread["key"],
            "game_day": thread["game_day"],
            "opponent": thread["opponent"],
            "articles": [public(article) for article in thread["articles"]],
        })

    reasons["selected_game_stories"] = sum(len(thread["articles"]) for thread in threads)
    reasons["selected_game_threads"] = len(threads)
    return {
        "generated_at": now.isoformat(),
        "hours_back": hours_back,
        "cutoff": cutoff.isoformat(),
        "reasons": dict(reasons),
        "timestamp_enrichment_attempts": timestamp_enrichments,
        "threads": public_threads,
        "diagnostics": diagnostics,
    }
