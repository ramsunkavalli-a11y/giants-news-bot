from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from dateutil import parser as dtparser

from v2_authors import is_core_game_writer
from v2_mlb_schedule import fetch_giants_schedule
from v2_story import candidate_preference_key

PACIFIC = ZoneInfo("America/Los_Angeles")
BASEBALL_DAY_SHIFT_HOURS = 12
DEFAULT_GAME_HOURS_BACK = 30
SCHEDULE_MATCH_MAX_HOURS = 48

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

# These phrases describe a game or game analysis directly. They are safe to use
# in either a headline or a structured summary.
GAME_ANALYSIS_PATTERNS = (
    "what we learned",
    "observations",
    "takeaways:",
    "takeaways ",
    "go-ahead",
    "earns win",
    "earned win",
    "quality start",
    "solid start",
    "sharp start",
    "sterling start",
    "strong start",
)

# Result language is useful when it is in the headline, where it describes the
# primary subject. We intentionally do not use these generic result phrases from
# a summary alone; reaction/commentary stories often mention the just-finished
# score in their dek even when the article is really about quotes or criticism.
TITLE_RESULT_PATTERNS = (
    " in win",
    " in loss",
    " win over ",
    " loss to ",
    " lost to ",
    " loses to ",
    " lose to ",
    " falls to ",
    " fall to ",
    " fell to ",
    " doom ",
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

# Common completed-game headline constructions that do not literally say
# "win" or "loss". Requiring a recognized opponent keeps phrases such as
# "blown out by criticism" from being treated as baseball results.
COMPLETED_GAME_RESULT = re.compile(
    r"\b(?:blown out|shut out|blanked|walked off|routed|trounced|crushed|"
    r"outslugged|outlasted|swept)\b.*\b(?:by|to)\b",
    flags=re.I,
)

# When a headline is explicitly about somebody's reaction, criticism, or
# comments, the game is context rather than the primary subject. This guard is
# applied only to inferred game stories; explicit upstream game-story metadata
# remains authoritative.
REACTION_TITLE_PATTERNS = (
    " reacts to ",
    " reaction to ",
    " calls out ",
    " sounds off ",
    " weighs in ",
    " reveals message ",
    " team meeting after ",
)
REACTION_VERB_TARGET = re.compile(
    r"\b(?:rip|rips|ripped|criticize|criticizes|criticized|question|questions|questioned)\b"
    r".{0,45}\b(?:giants|players?|team|club|teammates?|effort|focus|concentration|decision)\b",
    flags=re.I,
)
URL_GAME_RESULT = re.compile(
    r"\bgiants\b.*\b(?:beat|beats|win|won|lose|loses|lost|fall|falls|fell|defeat|defeats)\b",
    flags=re.I,
)
SEASON_OUTLOOK_RE = re.compile(
    r"\b\d{2,3}-loss season\b|\bseason looms\b|\bseason outlook\b",
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


def _article_dt(article: dict) -> datetime | None:
    value = article.get("_published_dt")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return _parse_dt(str(article.get("published", "") or ""))


def _looks_like_reaction_title(title: str) -> bool:
    lower = f" {title.lower()} "
    return (
        any(pattern in lower for pattern in REACTION_TITLE_PATTERNS)
        or bool(REACTION_VERB_TARGET.search(title))
    )


def is_game_story(article: dict) -> bool:
    if article.get("content_type") == "game_story":
        return True
    if article.get("quality_reason") == "game_story_or_postgame_analysis":
        return True

    title = str(article.get("title", "") or "")
    summary = str(article.get("summary", "") or "")
    title_lower = title.lower()
    summary_lower = summary.lower()

    # A season outlook may use words such as "loss" but is not coverage of
    # the just-finished game and belongs in the main feed.
    if SEASON_OUTLOOK_RE.search(title):
        return False

    # MLB's official recap headlines are often colorful rather than result-led,
    # while the article slug retains the concrete game outcome. Use that
    # publisher metadata only when it contains an explicit result construction.
    url_slug = urlparse(str(article.get("url", "") or "")).path.rsplit("/", 1)[-1]
    if article.get("source") == "MLB.com" and (
        RESULT_VERBS.search(url_slug.replace("-", " "))
        or GIANTS_RESULT.search(url_slug.replace("-", " "))
        or URL_GAME_RESULT.search(url_slug.replace("-", " "))
    ):
        return True

    headline_game_signal = (
        any(pattern in title_lower for pattern in GAME_ANALYSIS_PATTERNS)
        or any(pattern in title_lower for pattern in TITLE_RESULT_PATTERNS)
        or bool(RESULT_VERBS.search(title))
        or bool(GIANTS_RESULT.search(title))
        or (
            bool(COMPLETED_GAME_RESULT.search(title))
            and bool(extract_opponent({"title": title}))
        )
    )
    if headline_game_signal:
        return not _looks_like_reaction_title(title)

    # A summary may rescue a headline-framed analysis story when it contains a
    # concrete on-field/game-analysis signal (for example, "quality start").
    # Generic result language in a summary is deliberately insufficient.
    return any(pattern in summary_lower for pattern in GAME_ANALYSIS_PATTERNS)


def baseball_day(article: dict) -> str:
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


def game_thread_key(game_day: str, opponent: str = "", game_pk: int = 0) -> str:
    if game_pk:
        return f"game:{game_pk}"
    return f"game:{game_day}:{opponent or 'unknown'}"


def order_game_articles(members: list[dict]) -> list[dict]:
    core = [article for article in members if is_core_game_writer(article.get("author", ""))]
    if not core:
        return sorted(members, key=candidate_preference_key, reverse=True)

    ceiling = datetime.max.replace(tzinfo=timezone.utc)
    root = min(
        core,
        key=lambda article: (
            _article_dt(article) or ceiling,
            article.get("url", ""),
        ),
    )
    replies = [article for article in members if article is not root]
    replies.sort(
        key=lambda article: (
            _article_dt(article) or ceiling,
            article.get("source", ""),
            article.get("url", ""),
        )
    )
    return [root, *replies]


def _schedule_games_for_articles(articles: list[dict]) -> list[dict]:
    dates = [
        dt.astimezone(PACIFIC).date()
        for article in articles
        if (dt := _article_dt(article)) is not None
    ]
    if not dates:
        return []
    try:
        return fetch_giants_schedule(min(dates) - timedelta(days=3), max(dates) + timedelta(days=1))
    except Exception:
        return []


def _match_schedule_game(article: dict, games: list[dict]) -> dict | None:
    article_dt = _article_dt(article)
    opponent = extract_opponent(article)
    if article_dt is None or not opponent:
        return None

    candidates: list[tuple[float, dict]] = []
    for game in games:
        if game.get("opponent") != opponent:
            continue
        game_dt = _parse_dt(str(game.get("game_date", "") or ""))
        if game_dt is None or game_dt > article_dt:
            continue
        age_hours = (article_dt - game_dt).total_seconds() / 3600
        if age_hours > SCHEDULE_MATCH_MAX_HOURS:
            continue
        candidates.append((age_hours, game))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], -int(item[1].get("game_pk") or 0)))
    return candidates[0][1]


def group_game_articles(
    articles: list[dict],
    *,
    schedule_games: list[dict] | None = None,
) -> list[dict]:
    """Group by real MLB game when supplied; otherwise use the legacy heuristic."""
    games = schedule_games or []
    grounded: dict[int, list[dict]] = defaultdict(list)
    fallback: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    game_meta: dict[int, dict] = {}

    for article in articles:
        game = _match_schedule_game(article, games)
        game_pk = int((game or {}).get("game_pk") or 0)
        if game_pk:
            grounded[game_pk].append(article)
            game_meta[game_pk] = game
            continue

        day = baseball_day(article)
        if not day:
            continue
        fallback[day][extract_opponent(article)].append(article)

    groups: list[dict] = []
    for game_pk, members in grounded.items():
        game = game_meta[game_pk]
        groups.append({
            "key": game_thread_key(game.get("official_date", ""), game.get("opponent", ""), game_pk),
            "game_pk": game_pk,
            "game_day": game.get("official_date", ""),
            "opponent": game.get("opponent", ""),
            "schedule_grounded": True,
            "articles": order_game_articles(members),
        })

    for day, opponent_groups in fallback.items():
        known = [opponent for opponent in opponent_groups if opponent]
        unknown = opponent_groups.pop("", [])
        if unknown and len(known) == 1:
            opponent_groups[known[0]].extend(unknown)
        elif unknown:
            opponent_groups[""] = unknown

        for opponent, members in opponent_groups.items():
            groups.append({
                "key": game_thread_key(day, opponent),
                "game_pk": 0,
                "game_day": day,
                "opponent": opponent,
                "schedule_grounded": False,
                "articles": order_game_articles(members),
            })

    def newest(group: dict) -> datetime:
        values = [_article_dt(item) for item in group["articles"]]
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
                article["content_type"] = "game_story"
                article["section"] = f"game_thread:{article.get('section', '')}"
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

    schedule_games = _schedule_games_for_articles(eligible)
    threads = group_game_articles(eligible, schedule_games=schedule_games)

    def public(article: dict) -> dict:
        return {key: value for key, value in article.items() if not key.startswith("_")}

    public_threads = []
    for thread in threads:
        public_threads.append({
            "key": thread["key"],
            "game_pk": thread.get("game_pk", 0),
            "game_day": thread["game_day"],
            "opponent": thread["opponent"],
            "schedule_grounded": thread.get("schedule_grounded", False),
            "articles": [public(article) for article in thread["articles"]],
        })

    reasons["schedule_grounded_threads"] = sum(1 for thread in threads if thread.get("schedule_grounded"))
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
