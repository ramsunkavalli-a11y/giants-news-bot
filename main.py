from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter

from bsky_client import build_post_text, bsky_login, post_to_bluesky
from config import GOOGLE_NEWS_SOURCES, NON_RSS_SOURCES, RSS_ONLY_NAMES, RSS_ONLY_SOURCES, Settings
from discovery_google import discover_google_news
from discovery_nonrss import discover_non_rss
from discovery_rss import discover_rss_sources
from filters import (
    canonicalize_url,
    clean_text,
    is_bad_domain,
    is_recent_enough,
    is_relevant_giants,
    is_story_url,
    source_url_allowed,
)
from models import Candidate
from parser_meta import extract_meta
from scoring import score_candidate
from state import now_utc, prune_state, state_load, state_save


class BudgetTracker:
    def __init__(self, max_requests: int, max_seconds: int):
        self.max_requests = max_requests
        self.max_seconds = max_seconds
        self.start = time.monotonic()
        self.requests = 0
        self._logged = False

    def discovery_seconds(self) -> float:
        return time.monotonic() - self.start

    def allow_request(self) -> bool:
        return self.requests < self.max_requests and self.discovery_seconds() < self.max_seconds

    def consume(self) -> None:
        self.requests += 1

    def budget_exceeded(self) -> bool:
        return not self.allow_request()

    def log_once(self, logger) -> None:
        if self._logged:
            return
        self._logged = True
        logger(
            f"budget_exceeded max_requests={self.max_requests} used_requests={self.requests} "
            f"max_seconds={self.max_seconds} elapsed={self.discovery_seconds():.1f}"
        )


class BudgetSession:
    def __init__(self, session: requests.Session, budget: BudgetTracker, logger):
        self._session = session
        self._budget = budget
        self._logger = logger

    def request(self, method: str, url: str, **kwargs):
        if self._budget.budget_exceeded():
            self._budget.log_once(self._logger)
            raise RuntimeError("request_budget_exceeded")
        self._budget.consume()
        return self._session.request(method, url, **kwargs)

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)

    def head(self, url: str, **kwargs):
        return self.request("HEAD", url, **kwargs)


def log(message: str) -> None:
    print(f"[{datetime.utcnow().isoformat()}] {message}")


def create_session(settings: Settings) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": settings.user_agent})
    session.mount("http://", HTTPAdapter(pool_connections=20, pool_maxsize=20))
    session.mount("https://", HTTPAdapter(pool_connections=20, pool_maxsize=20))
    return session


def resolve_final_url(session: BudgetSession, settings: Settings, url: str, redirect_cache: Dict[str, str]) -> str:
    c = canonicalize_url(url, settings)
    if not c:
        return ""
    if c in redirect_cache:
        return redirect_cache[c]
    final = c
    try:
        r = session.request("HEAD", c, timeout=settings.request_timeout, allow_redirects=True)
        final = canonicalize_url(r.url or c, settings)
    except Exception:
        final = c
    redirect_cache[c] = final
    return final


def fetch_html(session: BudgetSession, settings: Settings, url: str) -> str:
    try:
        r = session.get(url, timeout=settings.request_timeout)
        if r.status_code >= 400 or "text/html" not in r.headers.get("Content-Type", ""):
            return ""
        return r.text[:1_200_000]
    except Exception:
        return ""


def enrich_candidate(session: BudgetSession, settings: Settings, state: Dict[str, Any], c: Candidate) -> Candidate:
    cache = state["meta_cache"].get(c.url, {})
    if cache:
        c.title = c.title or cache.get("title", "")
        c.author = c.author or cache.get("author", "")
        c.summary = c.summary or cache.get("summary", "")
        c.image_url = c.image_url or cache.get("image_url", "")
        c.article_meta_confirmed = bool(cache.get("article_meta_confirmed", False))
        return c

    html = fetch_html(session, settings, c.url)
    if html:
        meta = extract_meta(c.url, html)
        if meta.canonical:
            c.url = resolve_final_url(session, settings, canonicalize_url(urljoin(c.url, meta.canonical), settings), state["redirect_cache"])
        c.title = c.title or meta.title
        c.author = c.author or meta.author
        c.summary = c.summary or meta.description
        c.image_url = c.image_url or canonicalize_url(urljoin(c.url, meta.image_url), settings)
        c.article_meta_confirmed = meta.article_meta_confirmed or (
            bool(c.title) and bool(c.summary) and bool(meta.canonical)
        )

    state["meta_cache"][c.url] = {
        "title": c.title,
        "author": c.author,
        "summary": c.summary,
        "image_url": c.image_url,
        "article_meta_confirmed": c.article_meta_confirmed,
        "ts": now_utc().isoformat(),
    }
    return c


def validate_candidates(
    session: BudgetSession,
    settings: Settings,
    state: Dict[str, Any],
    candidates: List[Candidate],
    diagnostics: Dict[str, Any],
    budget: BudgetTracker,
) -> List[Candidate]:
    out: List[Candidate] = []
    seen = set()
    enrich_count = 0

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

        c.url = resolve_final_url(session, settings, c.url, state["redirect_cache"])
        domain = urlparse(c.url).netloc.lower()
        if is_bad_domain(domain):
            diagnostics["rejections"][c.source]["bad_domain"] += 1
            continue
        if not source_url_allowed(c.source, c.url):
            reason = "mlb_non_article_url" if domain.endswith("mlb.com") else "source_url_rules"
            diagnostics["rejections"][c.source][reason] += 1
            continue
        if not is_story_url(c.url):
            diagnostics["rejections"][c.source]["not_story_url"] += 1
            continue

        if not is_relevant_giants(c.title, c.summary, c.categories, c.url):
            diagnostics["rejections"][c.source]["irrelevant_prefilter"] += 1
            continue

        if enrich_count >= settings.max_enrich_candidates:
            diagnostics["rejections"][c.source]["enrich_budget_exceeded"] += 1
            continue
        if budget.budget_exceeded():
            budget.log_once(log)
            diagnostics["rejections"][c.source]["request_budget_exceeded"] += 1
            continue

        enrich_count += 1
        c = enrich_candidate(session, settings, state, c)

        if not is_relevant_giants(c.title, c.summary, c.categories, c.url):
            diagnostics["rejections"][c.source]["irrelevant"] += 1
            continue

        c.title = clean_text(c.title) or "Giants update"
        c.author = clean_text(c.author)
        c.summary = clean_text(c.summary)

        # strict MLB rule: metadata should confirm article when available
        if urlparse(c.url).netloc.lower().endswith("mlb.com") and not c.article_meta_confirmed:
            diagnostics["rejections"][c.source]["mlb_missing_article_meta"] += 1
            continue

        c = score_candidate(c)
        if c.score < settings.score_threshold:
            diagnostics["rejections"][c.source]["low_score"] += 1
            continue

        if c.discovered_via == "rss":
            c.selected_reasons.append("rss")
        if c.article_meta_confirmed:
            c.selected_reasons.append("article_meta_confirmed")
        if c.author and "priority_author" in c.score_components:
            c.selected_reasons.append("priority_author")

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
        "article_meta_confirmed": c.article_meta_confirmed,
        "selected_reasons": c.selected_reasons,
    }


def select_candidates(settings: Settings, candidates: List[Candidate]) -> List[Candidate]:
    candidates.sort(
        key=lambda c: (
            1 if c.discovered_via == "rss" else 0,
            1 if c.article_meta_confirmed else 0,
            c.score,
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
    budget = BudgetTracker(settings.max_total_http_requests, settings.max_discovery_seconds)
    raw_session = create_session(settings)
    session = BudgetSession(raw_session, budget, log)

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
        log(
            f"rss_health source={src} url={health['feed_url']} bozo={health['bozo']} "
            f"bozo_exception={health.get('bozo_exception') or 'none'} "
            f"status={health['status']} entries={health['entry_count']}"
        )

    google_candidates = discover_google_news(settings, GOOGLE_NEWS_SOURCES)
    nonrss_candidates = discover_non_rss(settings, session, NON_RSS_SOURCES)

    for c in rss_candidates + google_candidates + nonrss_candidates:
        diagnostics["discovered"][c.source].append(_cand_dict(c))

    log(f"discovered_counts rss={len(rss_candidates)} google={len(google_candidates)} nonrss={len(nonrss_candidates)}")
    rss_only_seen_nonrss = {c.source for c in nonrss_candidates if c.source in RSS_ONLY_NAMES}
    if rss_only_seen_nonrss:
        log(f"warning rss_only_source_seen_in_nonrss={sorted(rss_only_seen_nonrss)}")

    validated = validate_candidates(session, settings, state, rss_candidates + google_candidates + nonrss_candidates, diagnostics, budget)
    selected = select_candidates(settings, validated)

    all_sources = sorted({c.source for c in rss_candidates + google_candidates + nonrss_candidates})
    for source in all_sources:
        discovered_n = len(diagnostics["discovered"][source])
        validated_n = len(diagnostics["validated"][source])
        rejects = diagnostics["rejections"][source]
        rejected_n = sum(rejects.values())
        top = ", ".join([f"{k}={v}" for k, v in sorted(rejects.items(), key=lambda kv: kv[1], reverse=True)[:4]]) or "none"
        log(f"source_summary source={source} discovered={discovered_n} rejected={rejected_n} validated={validated_n} top_reasons={top}")

    for c in selected:
        diagnostics["selected"].append(_cand_dict(c))
        reason_blob = ",".join(c.selected_reasons) if c.selected_reasons else "score"
        log(f"selected source={c.source} title={c.title} url={c.url} why={reason_blob}")

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

    did, jwt = bsky_login(raw_session, settings.bsky_pds, settings.bsky_identifier, settings.bsky_app_password, settings.request_timeout)
    for c in selected:
        post_to_bluesky(raw_session, c, settings.bsky_pds, did, jwt, settings.request_timeout)
        state["posted_urls"][c.url] = now_utc().isoformat()
        log(f"posted {c.url}")
        time.sleep(0.8)

    state_save(settings, state)


if __name__ == "__main__":
    main()
