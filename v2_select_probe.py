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
UA = "Mozilla/5.0 GiantsNewsBotV2Selector/0.2"
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
        kl = key.lower()
        if kl in TRACKING_KEYS or kl.startswith("utm_") or kl.startswith("mc_"):
            continue
        filtered.append((key, value))
    cleaned = parsed._replace(
        netloc=parsed.netloc.lower(),
        query=urlencode(filtered, doseq=True),
        fragment="",
    )
    return urlunparse(cleaned)


def parse_dt(value: str):
    if not value:
        return None
    try:
        dt = dtparser.parse(value)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def optional_page_metadata(url: str) -> tuple[str, str]:
    """Best-effort title/time enrichment; never establishes article validity."""
    try:
        response = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT, allow_redirects=True)
        soup = BeautifulSoup(response.text, "lxml")
        title = ""
        for attrs in (
            {"property": "og:title"},
            {"name": "twitter:title"},
        ):
            tag = soup.find("meta", attrs=attrs)
            if tag and tag.get("content"):
                title = tag.get("content", "").strip()
                break
        if not title and soup.title:
            title = soup.title.get_text(" ", strip=True)
        if title.lower() in {"just a moment...", "access denied", ""}:
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
    """Fallback fingerprint when an already-posted publisher blocks metadata."""
    try:
        parts = [p for p in urlparse(url).path.split("/") if p]
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
    records: list[tuple[str, datetime | None]] = []
    if isinstance(posted_raw, dict):
        iterable = posted_raw.items()
    elif isinstance(posted_raw, list):
        iterable = ((url, "") for url in posted_raw)
    else:
        iterable = []

    # Small grace period catches an event posted just before the candidate window.
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
    records.sort(key=lambda item: item[1] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
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
            history.append({"url": url, "title": title, "posted_at": dt.isoformat() if dt else ""})
    return history, enrichments


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
    posted_records = posted_url_records(posted_raw, cutoff)
    posted_story_history, history_enrichments = build_posted_story_history(posted_records)

    reasons = Counter()
    eligible: list[dict] = []
    diagnostics: list[dict] = []
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
            published = article.get("published", "")
            dt = parse_dt(published)
            # NBC's clean landing-page adapter does not currently expose time.
            if dt is None and source == "NBC Sports Bay Area" and timestamp_enrichments < 8:
                _, enriched = optional_page_metadata(url)
                timestamp_enrichments += 1
                if enriched:
                    article["published"] = enriched
                    dt = parse_dt(enriched)
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

    # Suppress later versions of an event that the production bot already posted,
    # even when the publisher URL differs.
    unposted_story_articles: list[dict] = []
    for article in eligible:
        match = next(
            (h for h in posted_story_history if same_story(article.get("title", ""), h.get("title", ""))),
            None,
        )
        if match:
            reasons["story_already_posted"] += 1
            diagnostics.append({
                "source": article.get("source", ""),
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "published": article.get("published", ""),
                "quality": article.get("quality", ""),
                "reason": "story_already_posted",
                "matched_posted_title": match.get("title", ""),
                "matched_posted_url": match.get("url", ""),
            })
        else:
            unposted_story_articles.append(article)

    clusters = cluster_articles(unposted_story_articles)
    clusters.sort(
        key=lambda c: c.newest_dt or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    selected: list[dict] = []
    used_sources: set[str] = set()
    cluster_summaries: list[dict] = []
    overflow: list[dict] = []

    for cluster in clusters:
        ranked = cluster.ranked()
        chosen = next((a for a in ranked if a.get("source") not in used_sources), None)
        if chosen is None:
            reasons["source_cap_cluster"] += 1
            for member in ranked:
                diagnostics.append({
                    "source": member.get("source", ""),
                    "title": member.get("title", ""),
                    "url": member.get("url", ""),
                    "reason": "source_cap_cluster",
                })
            continue

        summary = {
            "chosen_title": chosen.get("title", ""),
            "chosen_source": chosen.get("source", ""),
            "chosen_author": chosen.get("author", ""),
            "member_count": len(ranked),
            "alternatives": [
                {
                    "source": a.get("source", ""),
                    "author": a.get("author", ""),
                    "title": a.get("title", ""),
                    "author_preference": a.get("author_preference", ""),
                    "source_preference": a.get("source_preference", ""),
                }
                for a in ranked if a is not chosen
            ],
        }
        cluster_summaries.append(summary)

        for member in ranked:
            if member is chosen:
                continue
            reasons["story_duplicate"] += 1
            diagnostics.append({
                "source": member.get("source", ""),
                "title": member.get("title", ""),
                "url": member.get("url", ""),
                "reason": "story_duplicate",
                "chosen_source": chosen.get("source", ""),
                "chosen_title": chosen.get("title", ""),
            })

        if len(selected) < MAX_POSTS:
            chosen["selection_preference"] = list(candidate_preference_key(chosen)[:-1])
            selected.append(chosen)
            used_sources.add(chosen.get("source", ""))
        else:
            overflow.append(chosen)
            reasons["run_cap"] += 1

    reasons["selected"] = len(selected)

    def public_article(article: dict) -> dict:
        return {k: v for k, v in article.items() if not k.startswith("_")}

    payload = {
        "generated_at": now.isoformat(),
        "hours_back": HOURS_BACK,
        "cutoff": cutoff.isoformat(),
        "max_posts": MAX_POSTS,
        "production_posted_url_count": len(posted),
        "recent_posted_story_count": len(posted_story_history),
        "timestamp_enrichment_attempts": timestamp_enrichments,
        "history_enrichment_attempts": history_enrichments,
        "reasons": dict(reasons),
        "clusters": cluster_summaries,
        "selected": [public_article(a) for a in selected],
        "run_cap_overflow": [public_article(a) for a in overflow],
        "posted_story_history": posted_story_history,
        "diagnostics": diagnostics,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(
        f"SELECTION_SIM state_urls={len(posted)} history_stories={len(posted_story_history)} "
        f"cutoff={cutoff.isoformat()} clusters={len(clusters)} selected={len(selected)}"
    )
    for idx, article in enumerate(selected, start=1):
        byline = f" · {article.get('author')}" if article.get("author") else ""
        access = " ($)" if article.get("access") == "paywalled" else ""
        print(
            f"WOULD_POST {idx}: {article.get('source')}{access}{byline} | "
            f"{article.get('title')} | {article.get('published')} | {article.get('url')}"
        )
    print("REASONS " + json.dumps(dict(reasons), sort_keys=True))

    duplicate_clusters = [c for c in cluster_summaries if c["member_count"] > 1]
    if duplicate_clusters:
        print("DUPLICATE_CLUSTERS:")
        for cluster in duplicate_clusters:
            print(
                f"  CHOSE {cluster['chosen_source']} · {cluster['chosen_author']} | "
                f"{cluster['chosen_title']}"
            )
            for alt in cluster["alternatives"]:
                print(f"    OVER {alt['source']} · {alt['author']} | {alt['title']}")


if __name__ == "__main__":
    main()
