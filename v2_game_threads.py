from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from dateutil import parser as dtparser

PACIFIC = ZoneInfo("America/Los_Angeles")
BASEBALL_DAY_SHIFT_HOURS = 12

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
    return (
        article.get("content_type") == "game_story"
        or article.get("quality_reason") == "game_story_or_postgame_analysis"
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
                "articles": members,
            })

    def newest(group: dict) -> datetime:
        values = [_parse_dt(item.get("published", "")) for item in group["articles"]]
        return max((value for value in values if value is not None), default=datetime.min.replace(tzinfo=timezone.utc))

    groups.sort(key=newest, reverse=True)
    return groups
