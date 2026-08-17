from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparser

from v2_story import (
    candidate_preference_key,
    cluster_articles,
    event_tokens,
    same_story,
    story_role,
)

UA = "Mozilla/5.0 GiantsNewsBotV2Selector/0.5"
TIMEOUT = 15
TRACKING_KEYS = {
    "fbclid", "gclid", "ref", "refsrc", "mc_cid", "mc_eid", "igshid", "source"
}
LOW_VALUE_TITLE_RE = re.compile(r"\bhighlights\b", flags=re.I)
ROTATION_WINDOW_DAYS = 14
EARLY_REPORTING_EDGE_MINUTES = 90


def canonicalize_url(url: str) -> str:
    if not isinstance(url, str) or not url:
        return ""
    parsed = urlparse(url.strip())
    filtered = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        low = key.lower()
        if low in TRACKING_KEYS or low.startswith("utm_") or low.startswith("mc_"):
            continue
        filtered.append((key, value))
    return urlunparse(parsed._replace(
        netloc=parsed.netloc.lower(),
        query=urlencode(filtered, doseq=True),
        fragment="",
    ))


def parse_dt(value: str):
    if not value:
        return None
    try:
        dt = dtparser.parse(value, tzinfos={"UT": timezone.utc})
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def optional_page_metadata(url: str) -> tuple[str, str]:
    """Best-effort title/time enrichment; never establishes article validity."""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": UA},
            timeout=TIMEOUT,
            allow_redirects=True,
        )
        soup = BeautifulSoup(response.text, "lxml")
        title = ""
        for attrs in ({"property": "og:title"}, {"name": "twitter:title"}):
            tag = soup.find("meta", attrs=attrs)
            if tag and tag.get("content"):
                title = tag.get("content", "").strip()
                break
        if not title and soup.title:
            title = soup.title.get_text(" ", strip=True)
        if title.lower() in {"", "just a moment...", "access denied"}:
            title = ""

        published = ""
        for attrs in (
            {"property": "article:published_time"},
            {"name": "article:published_time"},
            {"name": "date"},
            {"name": "publish-date"},
            {"name": "pubdate"},
        ):
            tag = soup.find("meta", attrs=attrs)
            if tag and tag.get("content") and parse_dt(tag.get("content", "")):
                published = tag.get("content", "")
                break
        if not published:
            time_tag = soup.find("time", attrs={"datetime": True})
            if time_tag and parse_dt(time_tag.get("datetime", "")):
                published = time_tag.get("datetime", "")
        return title, published
    except Exception:
        return "", ""


def title_from_url_slug(url: str) -> str:
    try:
        parts = [part for part in urlparse(url).path.split("/") if part]
        if not parts:
            return ""
        slug = parts[-1]
        if re.fullmatch(r"\d+", slug) and len(parts) > 1:
            slug = parts[-2]
        slug = re.sub(r"\.(?:html?|php)$", "", slug, flags=re.I)
        slug = re.sub(r"[-_]?\d{5,}$", "", slug)
        slug = re.sub(r"[-_]+", " ", slug).strip()
        return slug if len(slug.split()) >= 3 else ""
    except Exception:
        return ""


def _posted_url_records(posted_raw, cutoff: datetime) -> list[tuple[str, datetime | None]]:
    if isinstance(posted_raw, dict):
        iterable = posted_raw.items()
    elif isinstance(posted_raw, list):
        iterable = ((url, "") for url in posted_raw)
    else:
        iterable = []

    records: list[tuple[str, datetime | None]] = []
    history_cutoff = cutoff - timedelta(hours=24)
    for url, raw_ts in iterable:
        if isinstance(raw_ts, dict):
            raw_ts = raw_ts.get("ts", "") or raw_ts.get("posted_at", "")
        dt = parse_dt(str(raw_ts or ""))
        if dt is not None and dt < history_cutoff:
            continue
        canonical = canonicalize_url(url)
        if canonical:
            records.append((canonical, dt))
    floor = datetime.min.replace(tzinfo=timezone.utc)
    records.sort(key=lambda item: item[1] or floor, reverse=True)
    return records


def recent_story_history(
    state: dict,
    cutoff: datetime,
    max_legacy_enrichments: int = 24,
) -> tuple[list[dict], int]:
    """Use stored V2 titles first; backfill legacy URL-only state during migration."""
    history_cutoff = cutoff - timedelta(hours=24)
    history: list[dict] = []
    known_urls: set[str] = set()

    for item in state.get("posted_stories", []) or []:
        if not isinstance(item, dict):
            continue
        posted_at = parse_dt(item.get("posted_at", ""))
        if posted_at is not None and posted_at < history_cutoff:
            continue
        title = (item.get("title") or "").strip()
        url = canonicalize_url(item.get("url", ""))
        if title:
            stored = {
                "title": title,
                "url": url,
                "source": item.get("source", ""),
                "author": item.get("author", ""),
                "posted_at": posted_at.isoformat() if posted_at else "",
            }
            stored["story_role"] = story_role(stored)
            history.append(stored)
            if url:
                known_urls.add(url)

    enrichments = 0
    for url, dt in _posted_url_records(state.get("posted_urls", {}), cutoff):
        if url in known_urls:
            continue
        title = title_from_url_slug(url)
        if enrichments < max_legacy_enrichments:
            meta_title, _ = optional_page_metadata(url)
            enrichments += 1
            if meta_title:
                title = meta_title
        if title:
            legacy = {
                "title": title,
                "url": url,
                "source": "",
                "author": "",
                "posted_at": dt.isoformat() if dt else "",
            }
            legacy["story_role"] = story_role(legacy)
            history.append(legacy)
    return history, enrichments


def _recent_source_counts(state: dict, now: datetime) -> Counter:
    cutoff = now - timedelta(days=ROTATION_WINDOW_DAYS)
    counts = Counter()
    for item in state.get("posted_stories", []) or []:
        if not isinstance(item, dict) or item.get("kind", "standalone") != "standalone":
            continue
        dt = parse_dt(item.get("posted_at", ""))
        if dt is not None and dt < cutoff:
            continue
        source = str(item.get("source", "") or "")
        if source:
            counts[source] += 1
    return counts


def _choose_news_winner(members: list[dict], source_counts: Counter) -> tuple[dict, bool]:
    """Rotate truly comparable event reporting; meaningful quality gaps still win."""
    ranked = sorted(members, key=candidate_preference_key, reverse=True)
    if len(ranked) < 2 or not any(event_tokens(item.get("title", "")) for item in ranked):
        return ranked[0], False

    top_author_rank = candidate_preference_key(ranked[0])[0]
    comparable = [
        item for item in ranked
        if candidate_preference_key(item)[0] >= top_author_rank - 1
    ]
    if len(comparable) < 2:
        return ranked[0], False

    chronological = sorted(
        (item for item in comparable if isinstance(item.get("_published_dt"), datetime)),
        key=lambda item: item["_published_dt"],
    )
    if len(chronological) >= 2:
        lead = chronological[1]["_published_dt"] - chronological[0]["_published_dt"]
        if lead >= timedelta(minutes=EARLY_REPORTING_EDGE_MINUTES):
            return chronological[0], False

    def rotation_key(item: dict) -> tuple:
        author_rank, source_rank, named, timestamp = candidate_preference_key(item)
        return (
            source_counts.get(item.get("source", ""), 0),
            -author_rank,
            -source_rank,
            -named,
            -timestamp,
        )

    chosen = min(comparable, key=rotation_key)
    return chosen, chosen is not ranked[0]


def _role_representatives(cluster, source_counts: Counter) -> tuple[list[dict], list[dict], bool]:
    by_role = {"news": [], "analysis": []}
    for member in cluster.members:
        by_role.setdefault(story_role(member), []).append(member)

    representatives: list[dict] = []
    duplicates: list[dict] = []
    rotation_applied = False

    news = by_role.get("news", [])
    if news:
        chosen, rotated = _choose_news_winner(news, source_counts)
        representatives.append(chosen)
        duplicates.extend(item for item in news if item is not chosen)
        rotation_applied = rotated

    analysis = by_role.get("analysis", [])
    if analysis:
        chosen_analysis = max(analysis, key=candidate_preference_key)
        representatives.append(chosen_analysis)
        duplicates.extend(item for item in analysis if item is not chosen_analysis)

    return representatives, duplicates, rotation_applied


def select_articles(
    articles: list[dict],
    state: dict,
    *,
    hours_back: int = 72,
    max_posts: int = 5,
    now: datetime | None = None,
) -> dict:
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours_back)
    posted_raw = state.get("posted_urls", {})
    if isinstance(posted_raw, dict):
        posted = {canonicalize_url(url) for url in posted_raw}
    elif isinstance(posted_raw, list):
        posted = {canonicalize_url(url) for url in posted_raw}
    else:
        posted = set()
    posted.discard("")

    history, history_enrichments = recent_story_history(state, cutoff)
    source_counts = _recent_source_counts(state, now)
    reasons = Counter()
    diagnostics: list[dict] = []
    eligible: list[dict] = []
    timestamp_enrichments = 0

    for raw in articles:
        article = dict(raw)
        source = article.get("source", "")
        title = article.get("title", "")
        url = article.get("url", "")
        canonical = canonicalize_url(url)
        reason = "eligible"

        # Safety boundary: discovery adapters should already classify commodity
        # highlight pages as low value, but do not let one through if a title
        # variant misses an adapter pattern.
        if LOW_VALUE_TITLE_RE.search(title or ""):
            reason = "quality_low"
        elif article.get("quality") != "high":
            reason = f"quality_{article.get('quality') or 'unknown'}"
        elif not canonical:
            reason = "missing_url"
        elif canonical in posted:
            reason = "already_posted"
        else:
            dt = parse_dt(article.get("published", ""))
            if dt is None and source == "NBC Sports Bay Area" and timestamp_enrichments < 8:
                _, published = optional_page_metadata(url)
                timestamp_enrichments += 1
                if published:
                    article["published"] = published
                    dt = parse_dt(published)
            if dt is None:
                reason = "missing_published"
            elif dt < cutoff:
                reason = "stale"
            else:
                article["_published_dt"] = dt
                article["canonical_url"] = canonical
                article["story_role"] = story_role(article)
                eligible.append(article)

        reasons[reason] += 1
        diagnostics.append({
            "source": source,
            "title": article.get("title", ""),
            "url": url,
            "published": article.get("published", ""),
            "quality": article.get("quality", ""),
            "reason": reason,
        })

    fresh: list[dict] = []
    for article in eligible:
        role = story_role(article)
        match = next((
            item for item in history
            if item.get("story_role", "news") == role
            and same_story(article.get("title", ""), item.get("title", ""))
        ), None)
        if match:
            reasons["story_already_posted"] += 1
            diagnostics.append({
                "source": article.get("source", ""),
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "story_role": role,
                "reason": "story_already_posted",
                "matched_posted_title": match.get("title", ""),
                "matched_posted_url": match.get("url", ""),
            })
        else:
            fresh.append(article)

    clusters = cluster_articles(fresh)
    floor = datetime.min.replace(tzinfo=timezone.utc)
    clusters.sort(key=lambda cluster: cluster.newest_dt or floor, reverse=True)

    selected: list[dict] = []
    overflow: list[dict] = []
    used_sources: set[str] = set()
    summaries: list[dict] = []

    for cluster in clusters:
        representatives, duplicates, rotated = _role_representatives(cluster, source_counts)
        if not representatives:
            continue

        summaries.append({
            "member_count": len(cluster.members),
            "rotation_applied": rotated,
            "representatives": [
                {
                    "role": story_role(item),
                    "source": item.get("source", ""),
                    "author": item.get("author", ""),
                    "title": item.get("title", ""),
                }
                for item in representatives
            ],
            "alternatives": [
                {
                    "role": story_role(item),
                    "source": item.get("source", ""),
                    "author": item.get("author", ""),
                    "title": item.get("title", ""),
                }
                for item in duplicates
            ],
        })

        for duplicate in duplicates:
            reasons["story_duplicate"] += 1
            diagnostics.append({
                "source": duplicate.get("source", ""),
                "title": duplicate.get("title", ""),
                "url": duplicate.get("url", ""),
                "story_role": story_role(duplicate),
                "reason": "story_duplicate",
            })

        for chosen in representatives:
            source = chosen.get("source", "")
            if source in used_sources:
                reasons["source_cap_cluster"] += 1
                diagnostics.append({
                    "source": source,
                    "title": chosen.get("title", ""),
                    "url": chosen.get("url", ""),
                    "reason": "source_cap_cluster",
                })
                continue
            if len(selected) >= max_posts:
                overflow.append(chosen)
                reasons["run_cap"] += 1
                continue

            chosen["selection_preference"] = list(candidate_preference_key(chosen)[:-1])
            chosen["story_role"] = story_role(chosen)
            selected.append(chosen)
            used_sources.add(source)
            source_counts[source] += 1
            if rotated and story_role(chosen) == "news":
                reasons["comparable_story_rotation"] += 1

    reasons["selected"] = len(selected)

    def public(article: dict) -> dict:
        return {key: value for key, value in article.items() if not key.startswith("_")}

    return {
        "generated_at": now.isoformat(),
        "hours_back": hours_back,
        "cutoff": cutoff.isoformat(),
        "max_posts": max_posts,
        "production_posted_url_count": len(posted),
        "recent_posted_story_count": len(history),
        "recent_source_counts": dict(source_counts),
        "timestamp_enrichment_attempts": timestamp_enrichments,
        "history_enrichment_attempts": history_enrichments,
        "reasons": dict(reasons),
        "clusters": summaries,
        "selected": [public(article) for article in selected],
        "run_cap_overflow": [public(article) for article in overflow],
        "posted_story_history": history,
        "diagnostics": diagnostics,
    }
