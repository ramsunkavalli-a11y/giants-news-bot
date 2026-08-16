from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparser

HOURS_BACK = 72
MAX_POSTS = 5
UA = "Mozilla/5.0 GiantsNewsBotV2Selector/0.1"
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


def optional_published_time(url: str) -> str:
    """Best-effort timestamp enrichment; never establishes article validity."""
    try:
        response = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT, allow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        for attrs in (
            {"property": "article:published_time"},
            {"name": "article:published_time"},
            {"name": "date"},
            {"name": "publish-date"},
            {"name": "pubdate"},
        ):
            tag = soup.find("meta", attrs=attrs)
            if tag and tag.get("content") and parse_dt(tag.get("content", "")):
                return tag.get("content", "")
        time_tag = soup.find("time", attrs={"datetime": True})
        if time_tag and parse_dt(time_tag.get("datetime", "")):
            return time_tag.get("datetime", "")
    except Exception:
        pass
    return ""


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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
    reasons = Counter()
    eligible_by_source: dict[str, list[dict]] = {}
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
            # Enrich only the first few otherwise-eligible blank timestamps.
            if dt is None and source == "NBC Sports Bay Area" and timestamp_enrichments < 8:
                enriched = optional_published_time(url)
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
                eligible_by_source.setdefault(source, []).append(article)

        reasons[reason] += 1
        diagnostics.append({
            "source": source,
            "title": article.get("title", ""),
            "url": url,
            "published": article.get("published", ""),
            "quality": article.get("quality", ""),
            "reason": reason,
        })

    source_winners = []
    for source, items in eligible_by_source.items():
        items.sort(key=lambda x: x["_published_dt"], reverse=True)
        winner = items[0]
        source_winners.append(winner)
        for loser in items[1:]:
            reasons["source_cap"] += 1
            diagnostics.append({
                "source": source,
                "title": loser.get("title", ""),
                "url": loser.get("url", ""),
                "published": loser.get("published", ""),
                "quality": loser.get("quality", ""),
                "reason": "source_cap",
            })

    source_winners.sort(key=lambda x: x["_published_dt"], reverse=True)
    selected = source_winners[:MAX_POSTS]
    overflow = source_winners[MAX_POSTS:]
    reasons["selected"] = len(selected)
    reasons["run_cap"] += len(overflow)

    def public_article(article: dict) -> dict:
        out = {k: v for k, v in article.items() if not k.startswith("_")}
        return out

    payload = {
        "generated_at": now.isoformat(),
        "hours_back": HOURS_BACK,
        "cutoff": cutoff.isoformat(),
        "max_posts": MAX_POSTS,
        "production_posted_url_count": len(posted),
        "timestamp_enrichment_attempts": timestamp_enrichments,
        "reasons": dict(reasons),
        "source_winners": [public_article(a) for a in source_winners],
        "selected": [public_article(a) for a in selected],
        "run_cap_overflow": [public_article(a) for a in overflow],
        "diagnostics": diagnostics,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(
        f"SELECTION_SIM state_urls={len(posted)} cutoff={cutoff.isoformat()} "
        f"source_winners={len(source_winners)} selected={len(selected)}"
    )
    for idx, article in enumerate(selected, start=1):
        byline = f" · {article.get('author')}" if article.get("author") else ""
        access = " ($)" if article.get("access") == "paywalled" else ""
        print(
            f"WOULD_POST {idx}: {article.get('source')}{access}{byline} | "
            f"{article.get('title')} | {article.get('published')} | {article.get('url')}"
        )
    if overflow:
        print("RUN_CAP_OVERFLOW:")
        for article in overflow:
            print(f"  {article.get('source')} | {article.get('title')}")
    print("REASONS " + json.dumps(dict(reasons), sort_keys=True))


if __name__ == "__main__":
    main()
