from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparser

from v2_story import candidate_preference_key, cluster_articles, same_story

UA = "Mozilla/5.0 GiantsNewsBotV2Selector/0.4"
TIMEOUT = 15
TRACKING_KEYS = {
    "fbclid", "gclid", "ref", "refsrc", "mc_cid", "mc_eid", "igshid", "source"
}


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
            history.append({
                "title": title,
                "url": url,
                "posted_at": posted_at.isoformat() if posted_at else "",
            })
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
            history.append({
                "title": title,
                "url": url,
                "posted_at": dt.isoformat() if dt else "",
            })
    return history, enrichments


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
    reasons = Counter()
    diagnostics: list[dict] = []
    eligible: list[dict] = []
    timestamp_enrichments = 0

    for raw in articles:
        article = dict(raw)
        source = article.get("source", "")
        url = article.get("url", "")
        canonical = canonicalize_url(url)
        reason = "eligible"

        if article.get("quality") != "high":
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
        match = next((
            item for item in history
            if same_story(article.get("title", ""), item.get("title", ""))
        ), None)
        if match:
            reasons["story_already_posted"] += 1
            diagnostics.append({
                "source": article.get("source", ""),
                "title": article.get("title", ""),
                "url": article.get("url", ""),
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
        ranked = cluster.ranked()
        if not ranked:
            continue
        chosen = ranked[0]
        alternatives = ranked[1:]
        summaries.append({
            "chosen_title": chosen.get("title", ""),
            "chosen_source": chosen.get("source", ""),
            "chosen_author": chosen.get("author", ""),
            "member_count": len(ranked),
            "alternatives": [
                {
                    "source": item.get("source", ""),
                    "author": item.get("author", ""),
                    "title": item.get("title", ""),
                    "author_preference": item.get("author_preference", ""),
                    "source_preference": item.get("source_preference", ""),
                }
                for item in alternatives
            ],
        })
        for duplicate in alternatives:
            reasons["story_duplicate"] += 1
            diagnostics.append({
                "source": duplicate.get("source", ""),
                "title": duplicate.get("title", ""),
                "url": duplicate.get("url", ""),
                "reason": "story_duplicate",
                "chosen_source": chosen.get("source", ""),
                "chosen_title": chosen.get("title", ""),
            })

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
        selected.append(chosen)
        used_sources.add(source)

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
        "timestamp_enrichment_attempts": timestamp_enrichments,
        "history_enrichment_attempts": history_enrichments,
        "reasons": dict(reasons),
        "clusters": summaries,
        "selected": [public(article) for article in selected],
        "run_cap_overflow": [public(article) for article in overflow],
        "posted_story_history": history,
        "diagnostics": diagnostics,
    }
