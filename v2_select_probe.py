from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparser

from v2_story import candidate_preference_key, cluster_articles, same_story

HOURS_BACK = 72
MAX_POSTS = 5
MAX_HISTORY_ENRICHMENTS = 24
UA = "Mozilla/5.0 GiantsNewsBotV2Selector/0.3"
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
    """Best-effort title/time enrichment; failure never establishes invalidity."""
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
    """Fallback event fingerprint for publishers that block metadata requests."""
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


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def posted_url_records(posted_raw, cutoff: datetime) -> list[tuple[str, datetime | None]]:
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


def build_posted_story_history(records: list[tuple[str, datetime | None]]) -> tuple[list[dict], int]:
    history: list[dict] = []
    enrichments = 0
    for url, dt in records:
        title = title_from_url_slug(url)
        if enrichments < MAX_HISTORY_ENRICHMENTS:
            meta_title, _ = optional_page_metadata(url)
            enrichments += 1
            if meta_title:
                title = meta_title
        if title:
            history.append({
                "url": url,
                "title": title,
                "posted_at": dt.isoformat() if dt else "",
            })
    return history, enrichments


def public_article(article: dict) -> dict:
    return {key: value for key, value in article.items() if not key.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default="v2-probe.json")
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", default="v2-selection.json")
    args = parser.parse_args()

    probe = load_json(args.probe)
    try:
        state = load_json(args.state)
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f"production state unavailable or invalid: {exc}")

    posted_raw = state.get("posted_urls", {})
    if isinstance(posted_raw, dict):
        posted = {canonicalize_url(url) for url in posted_raw}
    elif isinstance(posted_raw, list):
        posted = {canonicalize_url(url) for url in posted_raw}
    else:
        posted = set()
    posted.discard("")

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=HOURS_BACK)
    records = posted_url_records(posted_raw, cutoff)
    posted_history, history_enrichments = build_posted_story_history(records)

    reasons = Counter()
    diagnostics: list[dict] = []
    eligible: list[dict] = []
    timestamp_enrichments = 0

    for raw in probe.get("articles", []):
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

    # URL dedupe is not enough: suppress a later publisher version when the
    # underlying event was already posted in the recent history window.
    fresh_articles: list[dict] = []
    for article in eligible:
        match = next((
            item for item in posted_history
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
            fresh_articles.append(article)

    clusters = cluster_articles(fresh_articles)
    floor = datetime.min.replace(tzinfo=timezone.utc)
    clusters.sort(key=lambda cluster: cluster.newest_dt or floor, reverse=True)

    selected: list[dict] = []
    overflow: list[dict] = []
    used_sources: set[str] = set()
    cluster_summaries: list[dict] = []

    for cluster in clusters:
        ranked = cluster.ranked()
        if not ranked:
            continue

        # Editorial quality wins before diversity. Never fall through to a worse
        # duplicate just because the best outlet already won another story.
        chosen = ranked[0]
        alternatives = ranked[1:]
        summary = {
            "chosen_title": chosen.get("title", ""),
            "chosen_source": chosen.get("source", ""),
            "chosen_author": chosen.get("author", ""),
            "member_count": len(ranked),
            "alternatives": [
                {
                    "source": article.get("source", ""),
                    "author": article.get("author", ""),
                    "title": article.get("title", ""),
                    "author_preference": article.get("author_preference", ""),
                    "source_preference": article.get("source_preference", ""),
                }
                for article in alternatives
            ],
        }
        cluster_summaries.append(summary)

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

        if len(selected) >= MAX_POSTS:
            overflow.append(chosen)
            reasons["run_cap"] += 1
            continue

        chosen["selection_preference"] = list(candidate_preference_key(chosen)[:-1])
        selected.append(chosen)
        used_sources.add(source)

    reasons["selected"] = len(selected)

    payload = {
        "generated_at": now.isoformat(),
        "hours_back": HOURS_BACK,
        "cutoff": cutoff.isoformat(),
        "max_posts": MAX_POSTS,
        "production_posted_url_count": len(posted),
        "recent_posted_story_count": len(posted_history),
        "timestamp_enrichment_attempts": timestamp_enrichments,
        "history_enrichment_attempts": history_enrichments,
        "reasons": dict(reasons),
        "clusters": cluster_summaries,
        "selected": [public_article(article) for article in selected],
        "run_cap_overflow": [public_article(article) for article in overflow],
        "posted_story_history": posted_history,
        "diagnostics": diagnostics,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(
        f"SELECTION_SIM state_urls={len(posted)} history_stories={len(posted_history)} "
        f"cutoff={cutoff.isoformat()} clusters={len(clusters)} selected={len(selected)}"
    )
    for index, article in enumerate(selected, start=1):
        byline = f" · {article.get('author')}" if article.get("author") else ""
        access = " ($)" if article.get("access") == "paywalled" else ""
        print(
            f"WOULD_POST {index}: {article.get('source')}{access}{byline} | "
            f"{article.get('title')} | {article.get('published')} | {article.get('url')}"
        )
    print("REASONS " + json.dumps(dict(reasons), sort_keys=True))

    duplicate_clusters = [item for item in cluster_summaries if item["member_count"] > 1]
    if duplicate_clusters:
        print("DUPLICATE_CLUSTERS:")
        for item in duplicate_clusters:
            print(
                f"  CHOSE {item['chosen_source']} · {item['chosen_author']} | "
                f"{item['chosen_title']}"
            )
            for alternative in item["alternatives"]:
                print(
                    f"    OVER {alternative['source']} · {alternative['author']} | "
                    f"{alternative['title']}"
                )


if __name__ == "__main__":
    main()
