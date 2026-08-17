from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

# Deliberately event-level. Generic team/player words are not enough to make two
# articles duplicates; we want to suppress the same news event, not different
# analysis about the same person.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "giant", "giants", "in", "into", "is", "it", "its", "mlb", "of",
    "on", "san", "sf", "francisco", "that", "the", "their", "this", "to", "with",
    "after", "before", "amid", "again", "new", "latest", "looking", "look", "toward",
}

EVENT_TOKENS = {
    "allstar", "surgery", "injury", "il", "retire", "trade", "promote", "callup",
    "suspend", "extension", "sign", "waiver", "dfa", "release", "hire", "fire",
    "host", "fracture", "rehab", "return", "debut", "roster", "deadline",
}

PHRASE_REPLACEMENTS = (
    (r"all[- ]star", "allstar"),
    (r"season[- ]ending", "seasonending"),
    (r"called? up", "callup"),
    (r"call[- ]up", "callup"),
    (r"gets? (?:the )?call(?: to (?:the )?(?:majors|big leagues))?", "callup"),
    (r"recalled? from (?:triple[- ]a|aaa)", "callup"),
    (r"placed on (?:the )?il", " il "),
    (r"designated for assignment", " dfa "),
)

TOKEN_ALIASES = {
    "retirement": "retire", "retires": "retire", "retired": "retire", "retiring": "retire",
    "traded": "trade", "trades": "trade", "trading": "trade", "acquire": "trade",
    "acquires": "trade", "acquired": "trade", "deal": "trade",
    "promoted": "promote", "promotes": "promote", "promotion": "promote",
    "signed": "sign", "signs": "sign", "signing": "sign",
    "suspended": "suspend", "suspends": "suspend", "suspension": "suspend",
    "fractured": "fracture", "fractures": "fracture",
    "rehabbing": "rehab", "rehabilitation": "rehab",
    "returns": "return", "returned": "return", "returning": "return",
    "debuted": "debut", "debuts": "debut",
    "hosts": "host", "hosting": "host", "hosted": "host",
    "injured": "injury", "injuries": "injury",
}

AUTHOR_RANK = {
    "elite": 5,
    "very_good": 4,
    "national": 4,
    "good": 3,
    "fine": 2,
    "": 1,
}

SOURCE_RANK = {
    "secondary": 0,
    "": 1,
}

ANALYSIS_AUTHORS = {
    "grant brisbee",
}
ANALYSIS_SOURCES = {
    "FanGraphs",
}
ANALYSIS_TITLE_PATTERNS = (
    "analysis",
    "breakdown",
    "scouting",
    "repertoire",
    "pitch mix",
    "pitching style",
    "chance to be",
    "what makes",
    "why ",
)


def _normalize_text(text: str) -> str:
    value = (text or "").lower().replace("’", "'")
    for pattern, replacement in PHRASE_REPLACEMENTS:
        value = re.sub(pattern, replacement, value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def story_tokens(title: str) -> set[str]:
    tokens: set[str] = set()
    for raw in _normalize_text(title).split():
        token = TOKEN_ALIASES.get(raw, raw)
        if token in STOPWORDS or len(token) < 2:
            continue
        tokens.add(token)
    return tokens


def event_tokens(title: str) -> set[str]:
    return story_tokens(title) & EVENT_TOKENS


def story_role(article: dict) -> str:
    """Distinguish deeper analysis from the event-reporting version of a story."""
    author = str(article.get("author", "") or "").strip().lower()
    source = str(article.get("source", "") or "")
    title = str(article.get("title", "") or "").lower()
    if author in ANALYSIS_AUTHORS or source in ANALYSIS_SOURCES:
        return "analysis"
    if any(pattern in title for pattern in ANALYSIS_TITLE_PATTERNS):
        return "analysis"
    return "news"


def same_story(title_a: str, title_b: str) -> bool:
    """Conservative event-level duplicate test for news headlines."""
    a = story_tokens(title_a)
    b = story_tokens(title_b)
    if not a or not b:
        return False
    overlap = a & b
    if len(overlap) < 2:
        return False

    events = (a & EVENT_TOKENS) & (b & EVENT_TOKENS)
    # One shared event concept + one shared identifying token (usually a player,
    # team action or distinctive year) is strong evidence of the same event.
    if events and len(overlap - EVENT_TOKENS) >= 1:
        return True

    union = a | b
    jaccard = len(overlap) / max(1, len(union))
    containment = len(overlap) / max(1, min(len(a), len(b)))
    # No explicit event word: require much stronger lexical agreement so two
    # separate analysis pieces about the same player do not collapse together.
    return len(overlap) >= 3 and (jaccard >= 0.48 or containment >= 0.72)


def candidate_preference_key(article: dict) -> tuple:
    """Best version of one event. Author dominates; source penalty is mild."""
    author = AUTHOR_RANK.get(article.get("author_preference", ""), 1)
    source = SOURCE_RANK.get(article.get("source_preference", ""), 1)
    dt = article.get("_published_dt")
    timestamp = dt.timestamp() if isinstance(dt, datetime) else 0.0
    # Prefer a named author over an anonymous candidate at otherwise equal tier.
    named = 1 if article.get("author") else 0
    return (author, source, named, timestamp)


@dataclass
class StoryCluster:
    members: list[dict] = field(default_factory=list)

    def matches(self, article: dict) -> bool:
        title = article.get("title", "")
        return any(same_story(title, member.get("title", "")) for member in self.members)

    def add(self, article: dict) -> None:
        self.members.append(article)

    @property
    def newest_dt(self):
        values = [m.get("_published_dt") for m in self.members if m.get("_published_dt")]
        return max(values) if values else None

    def ranked(self) -> list[dict]:
        return sorted(self.members, key=candidate_preference_key, reverse=True)


def cluster_articles(articles: Iterable[dict]) -> list[StoryCluster]:
    clusters: list[StoryCluster] = []
    ordered = sorted(
        articles,
        key=lambda a: a.get("_published_dt") or datetime.min,
        reverse=True,
    )
    for article in ordered:
        matched = None
        for cluster in clusters:
            if cluster.matches(article):
                matched = cluster
                break
        if matched is None:
            clusters.append(StoryCluster([article]))
        else:
            matched.add(article)
    return clusters
