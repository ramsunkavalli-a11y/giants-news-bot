from __future__ import annotations

import json
import time
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter

from bsky_client import build_post_text, bsky_login, post_to_bluesky
from config import GOOGLE_NEWS_SOURCES, NON_RSS_SOURCES, RSS_ONLY_NAMES, RSS_ONLY_SOURCES, Settings
from discovery_google import discover_google_news
from discovery_nonrss import discover_non_rss
from discovery_rss import discover_rss_sources
from filters import canonicalize_url, clean_text, is_bad_domain, is_recent_enough, is_relevant_giants, source_policy_allows, source_url_allowed
from models import Candidate
from scoring import score_candidate
from state import now_utc, prune_state, state_load, state_save
from url_resolver import resolve_article_url


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


def _cand_dict(c: Candidate) -> Dict[str, Any]:
    return {
        "source": c.source,
        "title": c.title,
        "feed_url": c.feed_url,
        "resolved_url": c.resolved_url,
        "publisher_url": c.publisher_url,
        "canonical_url": c.canonical_url,
        "post_url": c.post_url,
        "google_url": c.google_url,
        "validation_domain": c.validation_domain,
        "http_status": c.http_status,
        "content_type": c.content_type,
        "resolver_path": c.meta_sources_used[0] if c.meta_sources_used else "",
        "stage": c.stage,
        "skip_reason": c.skip_reason,
        "skip_reasons": c.reject_reasons,
        "source_policy_reason": c.source_policy_reason,
        "exception": c.exception,
        "is_cardable": c.is_cardable,
        "article_meta_confirmed": c.article_meta_confirmed,
        "score": c.score,
        "score_components": c.score_components,
        "selected_reasons": c.selected_reasons,
    }


def _reject(c: Candidate, diagnostics: Dict[str, Any], reason: str) -> None:
    c.add_reject(reason)
    c.stage = "skipped"
    diagnostics["rejections"][c.source][reason] += 1
    diagnostics["pipeline"].append(_cand_dict(c))


def _load_replay_candidates(path: str, limit: int) -> List[Candidate]:
    data = json.loads(Path(path).read_text())
    raw_items = data.get("pipeline") or []
    out: List[Candidate] = []
    for item in raw_items[:limit]:
        out.append(
            Candidate(
                source=item.get("source", "unknown"),
                url=item.get("feed_url") or item.get("url") or "",
                feed_url=item.get("feed_url") or item.get("url") or "",
                title=item.get("title", ""),
                summary=item.get("summary", ""),
                published_ts=item.get("published_ts", ""),
                discovered_via=item.get("discovered_via", "replay"),
            )
        )
    return out


def validate_candidates(
    session: BudgetSession,
    settings: Settings,
    state: Dict[str, Any],
    candidates: List[Candidate],
    diagnostics: Dict[str, Any],
    budget: BudgetTracker,
    verbose_resolver: bool,
) -> List[Candidate]:
    out: List[Candidate] = []
    seen = set()

    for c in candidates:
        c.stage = "discovered"
        c.feed_url = canonicalize_url(c.feed_url or c.url, settings)

        if not c.feed_url:
            _reject(c, diagnostics, "bad_feed_url")
            continue
        if not is_recent_enough(c.published_ts, settings.hours_back):
            _reject(c, diagnostics, "stale")
            continue

        res = resolve_article_url(c, session, settings, verbose=verbose_resolver)
        c.resolved_url = res.resolved_url
        c.publisher_url = res.resolved_url or c.resolved_url
        c.canonical_url = res.canonical_url
        c.post_url = res.post_url
        c.http_status = res.http_status
        c.content_type = res.content_type
        c.validation_domain = res.validation_domain
        c.article_meta_confirmed = res.article_meta_confirmed
        c.is_cardable = res.is_cardable
        c.exception = res.exception
        c.meta_sources_used = ([res.resolver_path] if res.resolver_path else []) + (res.meta_sources_used or [])

        if res.failure_reason == "unresolved_google_news_redirect":
            _reject(c, diagnostics, "unresolved_google_news_redirect")
            continue
        if not c.post_url:
            _reject(c, diagnostics, "unresolved_url")
            continue
        if "news.google.com" in c.post_url:
            _reject(c, diagnostics, "non_cardable_google_wrapper")
            continue

        dedupe_key = c.canonical_url or c.post_url
        if dedupe_key in state["posted_urls"] or dedupe_key in seen:
            _reject(c, diagnostics, "duplicate")
            continue

        if is_bad_domain(c.validation_domain):
            _reject(c, diagnostics, "bad_domain")
            continue

        validate_url = c.canonical_url or c.post_url
        if not source_url_allowed(c.source, validate_url):
            _reject(c, diagnostics, "source_url_rules")
            continue

        policy_ok, policy_reason = source_policy_allows(c.source, validate_url, c.title, c.summary)
        c.source_policy_reason = policy_reason
        if not policy_ok:
            _reject(c, diagnostics, policy_reason)
            continue

        c.url = validate_url
        c = score_candidate(c)
        c.stage = "scored"

        if c.score < settings.score_threshold:
            _reject(c, diagnostics, "low_provisional_score")
            continue

        if budget.budget_exceeded():
            budget.log_once(log)
            _reject(c, diagnostics, "request_budget_exceeded")
            continue

        if not is_relevant_giants(c.title, c.summary, c.categories, c.url):
            _reject(c, diagnostics, "irrelevant")
            continue

        c.title = clean_text(c.title) or "Giants update"
        c.author = clean_text(c.author)
        c.summary = clean_text(c.summary)

        c.is_cardable = c.post_url.startswith("https://") and "news.google.com" not in c.post_url
        if not c.is_cardable:
            _reject(c, diagnostics, "no_link_card")
            continue

        if c.google_url and c.post_url != c.google_url:
            c.selected_reasons.append("canonicalized_google_url")
        if c.article_meta_confirmed:
            c.selected_reasons.append("article_meta_confirmed")
        if c.discovered_via == "rss":
            c.selected_reasons.append("rss")

        c.stage = "validated"
        seen.add(dedupe_key)
        out.append(c)
        diagnostics["validated"][c.source].append(_cand_dict(c))
        diagnostics["pipeline"].append(_cand_dict(c))

    return out


def select_candidates(settings: Settings, candidates: List[Candidate]) -> List[Candidate]:
    candidates.sort(key=lambda c: (c.score, 1 if c.article_meta_confirmed else 0, c.published_ts), reverse=True)
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


def _diagnostics_summary(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    stage_counts: Dict[str, int] = defaultdict(int)
    resolver_counts: Dict[str, int] = defaultdict(int)
    source_counts: Dict[str, int] = defaultdict(int)
    skip_counts: Dict[str, int] = defaultdict(int)
    post_google_count = 0
    empty_resolved_count = 0

    samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    sample_reasons = {
        "unresolved_google_news_redirect",
        "non_cardable_google_wrapper",
        "blocked_ap_non_article",
        "low_provisional_score",
        "stale",
    }

    for item in diagnostics.get("pipeline", []):
        stage_counts[item.get("stage", "unknown")] += 1
        source_counts[item.get("source", "unknown")] += 1
        resolver = item.get("resolver_path", "")
        if resolver:
            resolver_counts[resolver] += 1
        reason = item.get("skip_reason", "")
        if reason:
            skip_counts[reason] += 1
            if reason in sample_reasons and len(samples[reason]) < 5:
                samples[reason].append(item)
        if item.get("http_status") == 200 and "news.google.com" in (item.get("post_url") or ""):
            post_google_count += 1
        if not item.get("resolved_url"):
            empty_resolved_count += 1

    return {
        "stage_counts": dict(stage_counts),
        "skip_reason_counts": dict(skip_counts),
        "per_source_counts": dict(source_counts),
        "resolver_strategy_counts": dict(resolver_counts),
        "http_200_post_url_google_count": post_google_count,
        "resolved_url_empty_count": empty_resolved_count,
        "failure_samples": dict(samples),
    }


def write_diagnostics(settings: Settings, payload: Dict[str, Any]) -> None:
    if not (settings.dry_run or settings.diagnostics_enabled):
        return
    payload["summary"] = _diagnostics_summary(payload)
    with open(settings.diagnostics_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def parse_args(argv: Optional[list[str]] = None) -> Namespace:
    p = ArgumentParser()
    p.add_argument("--diagnostics-in", default="")
    p.add_argument("--replay-candidates", action="store_true")
    p.add_argument("--limit", type=int, default=80)
    p.add_argument("--verbose-resolver", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
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

    if args.replay_candidates and args.diagnostics_in:
        candidates = _load_replay_candidates(args.diagnostics_in, args.limit)
    else:
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
        candidates = rss_candidates + google_candidates + nonrss_candidates

        log(f"discovered_counts rss={len(rss_candidates)} google={len(google_candidates)} nonrss={len(nonrss_candidates)}")
        rss_only_seen_nonrss = {c.source for c in nonrss_candidates if c.source in RSS_ONLY_NAMES}
        if rss_only_seen_nonrss:
            log(f"warning rss_only_source_seen_in_nonrss={sorted(rss_only_seen_nonrss)}")

    for c in candidates:
        diagnostics["discovered"][c.source].append(_cand_dict(c))

    validated = validate_candidates(session, settings, state, candidates[: args.limit], diagnostics, budget, args.verbose_resolver)
    selected = select_candidates(settings, validated)

    all_sources = sorted({c.source for c in candidates})
    for source in all_sources:
        discovered_n = len(diagnostics["discovered"][source])
        validated_n = len(diagnostics["validated"][source])
        rejects = diagnostics["rejections"][source]
        rejected_n = sum(rejects.values())
        top = ", ".join([f"{k}={v}" for k, v in sorted(rejects.items(), key=lambda kv: kv[1], reverse=True)[:5]]) or "none"
        log(f"source_summary source={source} discovered={discovered_n} rejected={rejected_n} validated={validated_n} top_reasons={top}")

    for c in selected:
        diagnostics["selected"].append(_cand_dict(c))
        reason_blob = ",".join(c.selected_reasons) if c.selected_reasons else "score"
        log(f"selected source={c.source} title={c.title} url={c.post_url or c.url} why={reason_blob}")

    write_diagnostics(settings, diagnostics)

    if not selected:
        log("No validated candidates selected.")
        state_save(settings, state)
        return

    if settings.dry_run:
        for c in selected:
            log(f"DRY_RUN would post: {build_post_text(c)}")
            state["posted_urls"][c.canonical_url or c.post_url or c.url] = now_utc().isoformat()
        state_save(settings, state)
        return

    if not settings.bsky_identifier or not settings.bsky_app_password:
        raise RuntimeError("BSKY_IDENTIFIER and BSKY_APP_PASSWORD are required when not DRY_RUN")

    did, jwt = bsky_login(raw_session, settings.bsky_pds, settings.bsky_identifier, settings.bsky_app_password, settings.request_timeout)
    for c in selected:
        post_to_bluesky(raw_session, c, settings.bsky_pds, did, jwt, settings.request_timeout)
        state["posted_urls"][c.canonical_url or c.post_url or c.url] = now_utc().isoformat()
        log(f"posted {c.post_url or c.url}")
        time.sleep(0.8)

    state_save(settings, state)


if __name__ == "__main__":
    main()
