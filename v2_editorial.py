"""Small editorial rules shared by discovery, routing and selection."""
from __future__ import annotations

import re
from urllib.parse import urlparse
from v2_authors import normalize_author

SEASON_REVIEW_RE = re.compile(
    r"\boff[- ]?season\b|\bseason (?:review|outlook|takeaways|in review)\b|"
    r"\b(?:takeaways|learned)\b.*\b(?:\d{4}|this|the|Giants['’]?) season\b|"
    r"\b\d{2,3}-loss season\b|\bseason looms\b",
    re.I,
)
RECURRING_CHAT_RE = re.compile(r"\bprospects chat:\s*\d{1,2}/\d{1,2}(?:/\d{4})?\b", re.I)
HYPOTHETICAL_TRADE_RE = re.compile(r"\b(?:mock trades?|trade proposals?|dream offseason)\b", re.I)
CONFIRMED_MOVE_RE = re.compile(
    r"\b(?:signs?|signed|re-signs?|re-signed|acquires?|acquired|trades?|traded|"
    r"claims?|claimed|waivers?|dfa|designated for assignment|releases?|released|"
    r"non-tenders?|non-tendered|tenders?|tendered|exercises?|exercised|declines?|declined|"
    r"agrees?|agreed|opts? out|opted out|hires?|hired|fires?|fired|promotes?|promoted|called up|surgery|injury|injured|"
    r"injuries|il|diagnosis|diagnosed)\b", re.I,
)
SPECULATIVE_RE = re.compile(
    r"\b(?:could|should|might|may|would|rumou?rs?|proposals?|targets?|fits?|"
    r"weighs?|considers?|pursues?|explores?|linked|interest)\b", re.I,
)
DEVELOPMENT_RE = re.compile(
    r"\b(?:prospects?|scouting|arbitration|free agen(?:t|ts|cy)|qualifying offer|"
    r"rule 5|40[- ]man|rotation|bullpen|payroll|roster|fall league|afl|"
    r"winter ball|winter meetings)\b", re.I,
)


def fangraphs_giants_evidence(article: dict) -> bool:
    """A team tag or Google query alone is insufficient for broad baseball posts.

    Inspect the headline, slug and beginning of the structured summary, never
    navigation/footer text or an author's identity. Mixed-team scouting is
    allowed when the structured item explicitly identifies Giants coverage.
    """
    text = " ".join((
        str(article.get("title", "") or ""),
        urlparse(str(article.get("url", "") or "")).path.replace("-", " "),
        str(article.get("summary", "") or "")[:600],
    ))
    return bool(re.search(r"\b(?:giants|san francisco)\b", text, re.I))


def is_confirmed_move(article: dict) -> bool:
    title = str(article.get("title", "") or "")
    return bool(CONFIRMED_MOVE_RE.search(title)) and not SPECULATIVE_RE.search(title)


def is_priority_author(article: dict) -> bool:
    return (article.get("source") == "The Athletic"
            and normalize_author(str(article.get("author", "") or "")) == "andrew baggarly")


def offseason_priority(article: dict) -> int:
    if article.get("_manual_priority"):
        return -1
    if is_priority_author(article):
        return 0
    if is_confirmed_move(article):
        return 0
    title = str(article.get("title", "") or "")
    if DEVELOPMENT_RE.search(title) or SEASON_REVIEW_RE.search(title):
        return 1
    return 2
