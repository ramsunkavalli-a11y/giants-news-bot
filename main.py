from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter

from bsky_client import build_post_text, bsky_login, post_to_bluesky
from config import GOOGLE_NEWS_SOURCES, NON_RSS_SOURCES, RSS_ONLY_NAMES, RSS_ONLY_SOURCES, Settings
from discovery_google import discover_google_news
from discovery_nonrss import discover_non_rss
from discovery_rss import discover_rss_sources
from filters import canonicalize_url, clean_text, is_bad_domain, is_recent_enough, is_relevant_giants, is_story_url
from models import Candidate
from parser_meta import extract_meta
from scoring import score_candidate
from state import now_utc, prune_state, state_load, state_save


def log(message: str) -> None:
    print(f"[{datetime.utcnow().isoformat()}] {message}")


def create_session(settings: Settings) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": settings.user_agent})
    session.mount("http://", HTTPAdapter(pool_connections=20, pool_maxsize=20))
    session.mount("https://", HTTPAdapter(pool_connections=20, pool_maxsize=20))
    return session


def resolve_final_url(session: requests.Session, settings: Settings, url: str, redirect_cache: Dict[str, str]) -> str:
    c = canonicalize_url(url, settings)
    if not c:
        return ""
    if c in redirect_cache:
        return redirect_cache[c]
    final = c
    try:
        r = session.request("HEAD", c, timeout=settings.request_timeout, allow_redirects=True)
        final = canonicalize_url(r.url or c, settings)
        if r.status_code >= 400:
            with session.get(c, timeout=settings.request_timeout, allow_redirects=True, stream=True) as fallback:
                final = canonicalize_url(fallback.url or c, settings)
    except Exception:
        pass
    redirect_cache[c] = final
    return final


def fetch_html(session: requests.Session, settings: Settings, url: str) -> str:
    try:
        r = session.get(url, timeout=settings.request_timeout)
        if r.status_code >= 400 or "text/html" not in r.headers.get("Content-Type", ""):
            return ""
        return r.text[:2_000_000]
    except Exception:
        return ""


def enrich_candidate(session: requests.Session, settings: Settings, state: Dict[str, Any], c: Candidate) -> Candidate:
    final = resolve_final_url(session, settings, c.url, state["redirect_cache"])
    c.url = final
    c.canonical_url = final
    cache = state["meta_cache"].get(final, {})

    if cache:
        c.title = c.title or cache.get("title", "")
        c.author = c.author or cache.get("author", "")
        c.summary = c.summary or cache.get("summary", "")
        c.image_url = c.image_url or cache.get("image_url", "")
        return c

    html = fetch_html(session, settings, final)
    if html:
        meta = extract_meta(final, html)
        if meta.canonical:
            c.url = resolve_final_url(session, settings, meta.canonical, state["redirect_cache"])
            c.canonical_url = c.url
        c.title = c.title or meta.title
        c.author = c.author or meta.author
        c.summary = c.summary or meta.description
        c.image_url = c.image_url or meta.image_url

    state["meta_cache"][c.url] = {
        "title": c.title,
        "author": c.author,
        "summary": c.summary,
        "image_url": c.image_url,
        "ts": now_utc().isoformat(),
    }
    return c


def validate_candidates(
    session: requests.Session,
    settings: Settings,
    state: Dict[str, Any],
    candidates: List[Candidate],
    diagnostics: Dict[str, Any],
) -> List[Candidate]:
    out: List[Candidate] = []
    seen = set()
    for c in candidates:
        raw_url = c.url
        c.url = canonicalize_url(c.url, settings)
        if not c.url:
            diagnostics["rejections"][c.source]["bad_url"] += 1
            continue
        if c.url in state["posted_urls"] or c.url in seen:
            diagnostics["rejections"][c.source]["duplicate"] += 1
            continue
        if not is_recent_enough(c.published_ts, settings.hours_back):
            diagnostics["rejections"][c.source]["stale"] += 1
            continue

        c = enrich_candidate(session, settings, state, c)
        domain = urlparse(c.url).netloc.lower()
        if is_bad_domain(domain):
            diagnostics["rejections"][c.source]["bad_domain"] += 1
            continue
        if not is_story_url(c.url):
            diagnostics["rejections"][c.source]["not_story_url"] += 1
            continue
        if not is_relevant_giants(c.title, c.summary, c.categories, c.url):
            diagnostics["rejections"][c.source]["irrelevant"] += 1
            continue

        c.title = clean_text(c.title) or "Giants update"
        c.author = clean_text(c.author)
        c.summary = clean_text(c.summary)
        c = score_candidate(c)
        if c.score < settings.score_threshold:
            diagnostics["rejections"][c.source]["low_score"] += 1
            continue

        seen.add(c.url)
        out.append(c)
        diagnostics["validated"][c.source].append(_cand_dict(c, raw_url=raw_url))
    return out


def _cand_dict(c: Candidate, raw_url: str = "") -> Dict[str, Any]:
    return {
        "source": c.source,
        "raw_url": raw_url or c.url,
        "url": c.url,
        "title": c.title,
        "author": c.author,
        "summary": c.summary,
        "published_ts": c.published_ts,
        "discovered_via": c.discovered_via,
        "score": c.score,
        "score_components": c.score_components,
    }


def select_candidates(settings: Settings, candidates: List[Candidate]) -> List[Candidate]:
    candidates.sort(
        key=lambda c: (
            c.score,
            1 if c.discovered_via == "rss" else 0,
            c.published_ts,
        ),
        reverse=True,
    )
    selected: List[Candidate] = []
    per_source: Dict[str, int] = defaultdict(int)
    for c in candidates:
        if len(selected) >= settings.max_posts_per_run:
            break
        if per_source[c.source] >= settings.max_per_source_per_run:
            continue
        selected.append(c)
        per_source[c.source] += 1
    return selected


def write_diagnostics(settings: Settings, payload: Dict[str, Any]) -> None:
    if not (settings.dry_run or settings.diagnostics_enabled):
        return
    with open(settings.diagnostics_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    settings = Settings()
    session = create_session(settings)
    state = state_load(settings)
    prune_state(settings, state)

    diagnostics: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": settings.dry_run,
        "sources": [s.__dict__ for s in (RSS_ONLY_SOURCES + GOOGLE_NEWS_SOURCES + NON_RSS_SOURCES)],
        "rss_health": {},
        "discovered": defaultdict(list),
        "validated": defaultdict(list),
        "rejections": defaultdict(lambda: defaultdict(int)),
        "selected": [],
    }

    rss_candidates, rss_health = discover_rss_sources(settings, RSS_ONLY_SOURCES)
    diagnostics["rss_health"] = rss_health
    for src, health in rss_health.items():
        log(f"rss_health source={src} url={health['feed_url']} bozo={health['bozo']} status={health['status']} entries={health['entry_count']}")

    google_candidates = discover_google_news(settings, GOOGLE_NEWS_SOURCES)
    nonrss_candidates = discover_non_rss(settings, session, NON_RSS_SOURCES)

    for c in rss_candidates + google_candidates + nonrss_candidates:
        diagnostics["discovered"][c.source].append(_cand_dict(c))

    log(f"discovered_counts rss={len(rss_candidates)} google={len(google_candidates)} nonrss={len(nonrss_candidates)}")
    rss_only_seen_nonrss = {c.source for c in nonrss_candidates if c.source in RSS_ONLY_NAMES}
    if rss_only_seen_nonrss:
        log(f"warning rss_only_source_seen_in_nonrss={sorted(rss_only_seen_nonrss)}")

    validated = validate_candidates(session, settings, state, rss_candidates + google_candidates + nonrss_candidates, diagnostics)
    selected = select_candidates(settings, validated)

    for c in selected:
        diagnostics["selected"].append(_cand_dict(c))
        log(
            f"selected source={c.source} via={c.discovered_via} score={c.score} "
            f"score_components={c.score_components} author={c.author or '<missing>'} title={c.title} url={c.url}"
        )

    for source, reasons in diagnostics["rejections"].items():
        if reasons:
            reason_blob = " ".join([f"{k}={v}" for k, v in sorted(reasons.items())])
            log(f"rejections source={source} {reason_blob}")

    write_diagnostics(settings, diagnostics)

    if not selected:
        log("No validated candidates selected.")
        state_save(settings, state)
        return

    if settings.dry_run:
        for c in selected:
            log(f"DRY_RUN would post: {build_post_text(c)}")
            state["posted_urls"][c.url] = now_utc().isoformat()
        state_save(settings, state)
        return

    if not settings.bsky_identifier or not settings.bsky_app_password:
        raise RuntimeError("BSKY_IDENTIFIER and BSKY_APP_PASSWORD are required when not DRY_RUN")

    did, jwt = bsky_login(session, settings.bsky_pds, settings.bsky_identifier, settings.bsky_app_password, settings.request_timeout)
    for c in selected:
        post_to_bluesky(session, c, settings.bsky_pds, did, jwt, settings.request_timeout)
        state["posted_urls"][c.url] = now_utc().isoformat()
        log(f"posted {c.url}")
        time.sleep(0.8)

    state_save(settings, state)


if __name__ == "__main__":
    main()
