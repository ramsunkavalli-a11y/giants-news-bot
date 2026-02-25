from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
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
    extract_external_url_from_text,
    extract_publisher_url_from_google_wrapper,
    is_bad_domain,
    is_recent_enough,
    is_relevant_giants,
    is_story_url,
    looks_like_google_wrapper,
    source_policy_allows,
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


def log(message: str) -> None:
    print(f"[{datetime.utcnow().isoformat()}] {message}")


def create_session(settings: Settings) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": settings.user_agent})
    session.mount("http://", HTTPAdapter(pool_connections=20, pool_maxsize=20))
    session.mount("https://", HTTPAdapter(pool_connections=20, pool_maxsize=20))
    return session


def pick_domain_for_validation(c: Candidate) -> str:
    for u in [c.canonical_url, c.publisher_url, c.resolved_url, c.feed_url, c.url]:
        if u:
            return urlparse(u).netloc.lower()
    return ""


def resolve_google_news_url(session: BudgetSession, settings: Settings, c: Candidate) -> None:
    raw = canonicalize_url(c.feed_url or c.url, settings)
    c.feed_url = raw
    c.url = raw
    c.stage = "canonicalized"

    if not raw:
        return
    if not looks_like_google_wrapper(raw):
        c.resolved_url = raw
        c.publisher_url = raw
        c.canonical_url = raw
        c.post_url = raw
        return

    c.google_url = raw

    # 1) try feed metadata/body extraction first
    hinted = extract_external_url_from_text(c.summary)
    if not hinted:
        hinted = extract_publisher_url_from_google_wrapper(raw)
    if hinted:
        c.resolved_url = canonicalize_url(hinted, settings)

    # 2) follow redirect chain from google wrapper
    try:
        with session.get(raw, timeout=settings.request_timeout, allow_redirects=True, stream=True) as r:
            redirect_final = canonicalize_url(r.url or "", settings)
            if redirect_final and "news.google.com" not in redirect_final:
                c.resolved_url = redirect_final
            c.http_status = r.status_code
            c.content_type = r.headers.get("Content-Type", "")
            html = ""
            if "text/html" in c.content_type:
                try:
                    html = r.text[:600_000]
                except Exception:
                    html = ""
            if html:
                meta = extract_meta(redirect_final or raw, html)
                cand = canonicalize_url(urljoin(redirect_final or raw, meta.canonical), settings) if meta.canonical else ""
                if cand and "news.google.com" not in cand:
                    c.publisher_url = cand
    except Exception as exc:
        c.exception = repr(exc)

    # 3) extra decode fallback if still unresolved
    if not c.resolved_url or "news.google.com" in c.resolved_url:
        fallback = extract_publisher_url_from_google_wrapper(raw)
        if fallback:
            c.resolved_url = canonicalize_url(fallback, settings)

    if not c.publisher_url:
        c.publisher_url = c.resolved_url or raw
    c.canonical_url = c.publisher_url
    c.post_url = c.publisher_url


def resolve_final_url(session: BudgetSession, settings: Settings, c: Candidate, redirect_cache: Dict[str, str]) -> None:
    target = canonicalize_url(c.publisher_url or c.resolved_url or c.feed_url or c.url, settings)
    if not target:
        return
    if target in redirect_cache:
        final = redirect_cache[target]
    else:
        final = target
        try:
            r = session.request("HEAD", target, timeout=settings.request_timeout, allow_redirects=True)
            final = canonicalize_url(r.url or target, settings)
        except Exception:
            pass
        redirect_cache[target] = final
    c.resolved_url = c.resolved_url or target
    c.publisher_url = final
    c.canonical_url = final
    c.post_url = final


def fetch_html(session: BudgetSession, settings: Settings, candidate: Candidate) -> str:
    try:
        r = session.get(candidate.post_url or candidate.publisher_url or candidate.url, timeout=settings.request_timeout)
        candidate.http_status = r.status_code
        candidate.content_type = r.headers.get("Content-Type", "")
        if r.status_code >= 400:
            candidate.add_reject("http_error")
            return ""
        if "text/html" not in candidate.content_type:
            candidate.add_reject("not_html")
            return ""
        candidate.stage = "fetched"
        return r.text[:1_200_000]
    except Exception as exc:
        candidate.exception = repr(exc)
        candidate.add_reject("fetch_exception")
        return ""


def enrich_candidate(session: BudgetSession, settings: Settings, state: Dict[str, Any], c: Candidate) -> Candidate:
    cache = state["meta_cache"].get(c.canonical_url or c.post_url or c.url, {})
    if cache:
        c.title = c.title or cache.get("title", "")
        c.author = c.author or cache.get("author", "")
        c.summary = c.summary or cache.get("summary", "")
        c.image_url = c.image_url or cache.get("image_url", "")
        c.article_meta_confirmed = bool(cache.get("article_meta_confirmed", False))
        c.meta_sources_used = list(cache.get("meta_sources_used", []))
        c.stage = "parsed"
        return c

    html = fetch_html(session, settings, c)
    if not html:
        return c

    meta = extract_meta(c.post_url or c.publisher_url or c.url, html)
    c.meta_sources_used = meta.meta_sources_used

    if meta.canonical:
        canon = canonicalize_url(urljoin(c.post_url or c.publisher_url or c.url, meta.canonical), settings)
        if canon:
            c.canonical_url = canon
            c.publisher_url = canon
            c.post_url = canon

    c.title = c.title or meta.title
    c.author = c.author or meta.author
    c.summary = c.summary or meta.description
    if meta.image_url:
        c.image_url = c.image_url or canonicalize_url(urljoin(c.post_url or c.publisher_url or c.url, meta.image_url), settings)

    c.article_meta_confirmed = meta.article_meta_confirmed or bool(c.title and c.post_url)
    c.is_cardable = bool(c.post_url and not looks_like_google_wrapper(c.post_url) and is_story_url(c.post_url))
    c.stage = "parsed"

    state["meta_cache"][c.canonical_url or c.post_url or c.url] = {
        "title": c.title,
        "author": c.author,
        "summary": c.summary,
        "image_url": c.image_url,
        "article_meta_confirmed": c.article_meta_confirmed,
        "meta_sources_used": c.meta_sources_used,
        "ts": now_utc().isoformat(),
    }
    return c


def _cand_dict(c: Candidate) -> Dict[str, Any]:
    return {
        "source": c.source,
        "feed_url": c.feed_url,
        "resolved_url": c.resolved_url,
        "publisher_url": c.publisher_url,
        "canonical_url": c.canonical_url,
        "post_url": c.post_url,
        "google_url": c.google_url,
        "validation_domain": c.validation_domain,
        "title": c.title,
        "author": c.author,
        "summary": c.summary,
        "published_ts": c.published_ts,
        "discovered_via": c.discovered_via,
        "stage": c.stage,
        "skip_reason": c.skip_reason,
        "skip_reasons": c.reject_reasons,
        "source_policy_reason": c.source_policy_reason,
        "exception": c.exception,
        "http_status": c.http_status,
        "content_type": c.content_type,
        "article_meta_confirmed": c.article_meta_confirmed,
        "is_cardable": c.is_cardable,
        "score": c.score,
        "score_components": c.score_components,
        "selected_reasons": c.selected_reasons,
        "meta_sources_used": c.meta_sources_used,
    }


def _reject(c: Candidate, diagnostics: Dict[str, Any], reason: str) -> None:
    c.add_reject(reason)
    c.stage = "skipped"
    diagnostics["rejections"][c.source][reason] += 1
    diagnostics["pipeline"].append(_cand_dict(c))


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
        c.stage = "discovered"
        c.feed_url = c.feed_url or c.url

        c.feed_url = canonicalize_url(c.feed_url, settings)
        if not c.feed_url:
            _reject(c, diagnostics, "bad_feed_url")
            continue
        if not is_recent_enough(c.published_ts, settings.hours_back):
            _reject(c, diagnostics, "stale")
            continue

        resolve_google_news_url(session, settings, c)
        if c.google_url and (not c.publisher_url or "news.google.com" in c.publisher_url):
            _reject(c, diagnostics, "unresolved_google_news_redirect")
            continue

        resolve_final_url(session, settings, c, state["redirect_cache"])

        dedupe_key = c.canonical_url or c.publisher_url or c.resolved_url or c.feed_url
        if not dedupe_key:
            _reject(c, diagnostics, "no_canonical_url")
            continue
        if dedupe_key in state["posted_urls"] or dedupe_key in seen:
            _reject(c, diagnostics, "duplicate")
            continue

        c.validation_domain = pick_domain_for_validation(c)
        if is_bad_domain(c.validation_domain):
            _reject(c, diagnostics, "bad_domain")
            continue

        validate_url = c.canonical_url or c.publisher_url or c.resolved_url or c.feed_url
        if not source_url_allowed(c.source, validate_url):
            _reject(c, diagnostics, "mlb_non_article_url" if c.validation_domain.endswith("mlb.com") else "source_url_rules")
            continue
        if not is_story_url(validate_url):
            _reject(c, diagnostics, "not_story_url")
            continue

        policy_ok, policy_reason = source_policy_allows(c.source, validate_url, c.title, c.summary)
        c.source_policy_reason = policy_reason
        if not policy_ok:
            _reject(c, diagnostics, policy_reason)
            continue

        c.url = validate_url
        c = score_candidate(c)
        c.stage = "scored"
        if c.score < 1:
            _reject(c, diagnostics, "low_provisional_score")
            continue

        if enrich_count >= settings.max_enrich_candidates:
            _reject(c, diagnostics, "enrich_budget_exceeded")
            continue
        if budget.budget_exceeded():
            budget.log_once(log)
            _reject(c, diagnostics, "request_budget_exceeded")
            continue

        enrich_count += 1
        c = enrich_candidate(session, settings, state, c)

        if c.discovered_via == "nonrss" and not c.title and not c.summary and not c.canonical_url:
            _reject(c, diagnostics, "empty_candidate")
            continue
        if c.discovered_via == "nonrss" and not c.title and not c.summary:
            _reject(c, diagnostics, "no_title_or_summary")
            continue

        if not is_relevant_giants(c.title, c.summary, c.categories, c.post_url or c.url):
            _reject(c, diagnostics, "irrelevant")
            continue

        policy_ok, policy_reason = source_policy_allows(c.source, c.post_url or c.url, c.title, c.summary)
        c.source_policy_reason = policy_reason
        if not policy_ok:
            _reject(c, diagnostics, policy_reason)
            continue

        if c.validation_domain.endswith("mlb.com") and not c.article_meta_confirmed:
            _reject(c, diagnostics, "mlb_missing_article_meta")
            continue

        c.title = clean_text(c.title) or "Giants update"
        c.author = clean_text(c.author)
        c.summary = clean_text(c.summary)

        c.url = c.post_url or c.canonical_url or c.publisher_url or c.resolved_url or c.feed_url
        c = score_candidate(c)
        c.stage = "validated"
        c.is_cardable = bool(c.url and not looks_like_google_wrapper(c.url))

        if not c.is_cardable:
            _reject(c, diagnostics, "no_link_card")
            continue
        if c.score < settings.score_threshold:
            _reject(c, diagnostics, "low_score")
            continue

        if c.discovered_via == "rss":
            c.selected_reasons.append("rss")
        if c.article_meta_confirmed:
            c.selected_reasons.append("article_meta_confirmed")
        if c.google_url and c.url != c.google_url:
            c.selected_reasons.append("canonicalized_google_url")
        if c.author and "priority_author" in c.score_components:
            c.selected_reasons.append("priority_author")

        seen.add(dedupe_key)
        out.append(c)
        diagnostics["validated"][c.source].append(_cand_dict(c))
        diagnostics["pipeline"].append(_cand_dict(c))
    return out


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
        c.stage = "selected"
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
        "pipeline": [],
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

    validated = validate_candidates(
        session,
        settings,
        state,
        rss_candidates + google_candidates + nonrss_candidates,
        diagnostics,
        budget,
    )
    selected = select_candidates(settings, validated)

    all_sources = sorted({c.source for c in rss_candidates + google_candidates + nonrss_candidates})
    for source in all_sources:
        discovered_n = len(diagnostics["discovered"][source])
        validated_n = len(diagnostics["validated"][source])
        rejects = diagnostics["rejections"][source]
        rejected_n = sum(rejects.values())
        top = ", ".join([f"{k}={v}" for k, v in sorted(rejects.items(), key=lambda kv: kv[1], reverse=True)[:5]]) or "none"
        log(
            f"source_summary source={source} discovered={discovered_n} "
            f"rejected={rejected_n} validated={validated_n} top_reasons={top}"
        )

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
            state["posted_urls"][c.canonical_url or c.url] = now_utc().isoformat()
        state_save(settings, state)
        return

    if not settings.bsky_identifier or not settings.bsky_app_password:
        raise RuntimeError("BSKY_IDENTIFIER and BSKY_APP_PASSWORD are required when not DRY_RUN")

    did, jwt = bsky_login(raw_session, settings.bsky_pds, settings.bsky_identifier, settings.bsky_app_password, settings.request_timeout)
    for c in selected:
        post_to_bluesky(raw_session, c, settings.bsky_pds, did, jwt, settings.request_timeout)
        state["posted_urls"][c.canonical_url or c.url] = now_utc().isoformat()
        log(f"posted {c.url}")
        time.sleep(0.8)

    state_save(settings, state)


if __name__ == "__main__":
    main()
