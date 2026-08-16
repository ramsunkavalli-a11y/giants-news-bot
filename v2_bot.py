from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup

from bsky_client import bsky_login, build_post_text, post_to_bluesky
from config import Settings
from models import Candidate
from v2_probe import (
    discover_athletic,
    discover_fangraphs,
    discover_mlb,
    discover_nbc,
    discover_sf_standard,
    discover_sfgate,
)
from v2_selector import canonicalize_url, parse_dt, select_articles

DISCOVERERS = [
    discover_sf_standard,
    discover_athletic,
    discover_mlb,
    discover_sfgate,
    discover_fangraphs,
    discover_nbc,
]


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {
            "posted_urls": {},
            "posted_stories": [],
            "redirect_cache": {},
            "meta_cache": {},
        }
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read().strip()
        state = json.loads(raw) if raw else {}
    except (OSError, json.JSONDecodeError):
        state = {}

    if not isinstance(state, dict):
        state = {}
    if not isinstance(state.get("posted_urls"), dict):
        state["posted_urls"] = {}
    if not isinstance(state.get("posted_stories"), list):
        state["posted_stories"] = []
    if not isinstance(state.get("redirect_cache"), dict):
        state["redirect_cache"] = {}
    if not isinstance(state.get("meta_cache"), dict):
        state["meta_cache"] = {}
    return state


def save_state(path: str, state: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True, ensure_ascii=False)


def prune_state(state: dict, keep_days: int) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
    posted = state.get("posted_urls", {})
    if isinstance(posted, dict):
        for url, raw_ts in list(posted.items()):
            if isinstance(raw_ts, dict):
                raw_ts = raw_ts.get("ts", "") or raw_ts.get("posted_at", "")
            dt = parse_dt(str(raw_ts or ""))
            if dt is not None and dt < cutoff:
                posted.pop(url, None)

    stories = []
    for item in state.get("posted_stories", []) or []:
        if not isinstance(item, dict):
            continue
        dt = parse_dt(item.get("posted_at", ""))
        if dt is None or dt >= cutoff:
            stories.append(item)
    state["posted_stories"] = stories


def discover_articles() -> tuple[list[dict], dict]:
    articles: list[dict] = []
    health: dict[str, dict] = {}
    for discover in DISCOVERERS:
        name = discover.__name__.replace("discover_", "")
        try:
            items = discover()
            health[name] = {"ok": True, "count": len(items)}
            articles.extend(asdict(item) for item in items)
        except Exception as exc:
            health[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    # Direct URL is the discovery identity in V2.
    unique = {item.get("url", ""): item for item in articles if item.get("url")}
    return list(unique.values()), health


def enrich_card_metadata(url: str, timeout: int = 15) -> dict:
    """Optional last-mile card enrichment. Failure must not block a selected story."""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 GiantsNewsBotV2Card/1.0"},
            timeout=timeout,
            allow_redirects=True,
        )
        soup = BeautifulSoup(response.text, "lxml")

        def meta(*pairs):
            for key, value in pairs:
                tag = soup.find("meta", attrs={key: value})
                if tag and tag.get("content"):
                    return tag.get("content", "").strip()
            return ""

        image = meta(("property", "og:image"), ("name", "twitter:image"))
        description = meta(
            ("property", "og:description"),
            ("name", "twitter:description"),
            ("name", "description"),
        )
        return {
            "ok": response.status_code < 400,
            "status": response.status_code,
            "image_url": image,
            "description": description,
        }
    except Exception as exc:
        return {"ok": False, "status": 0, "error": f"{type(exc).__name__}: {exc}"}


def article_to_candidate(article: dict, card_meta: dict) -> Candidate:
    url = (
        article.get("canonical_url")
        or canonicalize_url(article.get("url", ""))
        or article.get("url", "")
    )
    summary = article.get("summary", "") or card_meta.get("description", "")
    return Candidate(
        source=article.get("source", ""),
        url=url,
        title=article.get("title", ""),
        author=article.get("author", ""),
        summary=summary,
        image_url=card_meta.get("image_url", ""),
        discovered_via=article.get("section", "structured_v2"),
        published_ts=article.get("published", ""),
        access=article.get("access", "unknown"),
        resolved_url=url,
        publisher_url=url,
        canonical_url=url,
        post_url=url,
        article_meta_confirmed=bool(card_meta.get("ok")),
        is_cardable=True,
    )


def mark_posted(state: dict, article: dict) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    url = (
        article.get("canonical_url")
        or canonicalize_url(article.get("url", ""))
        or article.get("url", "")
    )
    state.setdefault("posted_urls", {})[url] = ts
    stories = state.setdefault("posted_stories", [])
    if not isinstance(stories, list):
        stories = []
        state["posted_stories"] = stories
    stories.append({
        "title": article.get("title", ""),
        "url": url,
        "source": article.get("source", ""),
        "author": article.get("author", ""),
        "posted_at": ts,
    })


def main() -> None:
    settings = Settings()
    state = load_state(settings.state_file)
    prune_state(state, settings.keep_posted_days)

    articles, health = discover_articles()
    selection = select_articles(
        articles,
        state,
        hours_back=settings.hours_back,
        max_posts=settings.max_posts_per_run,
    )

    candidates: list[tuple[dict, Candidate, dict]] = []
    for article in selection["selected"]:
        card_meta = enrich_card_metadata(article.get("url", ""), settings.request_timeout)
        candidate = article_to_candidate(article, card_meta)
        candidates.append((article, candidate, card_meta))

    diagnostics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": settings.dry_run,
        "health": health,
        "selection": selection,
        "posts": [
            {
                "text": build_post_text(candidate),
                "title": candidate.title,
                "summary": candidate.summary,
                "image_url": candidate.image_url,
                "url": candidate.post_url,
                "source": candidate.source,
                "author": candidate.author,
                "access": candidate.access,
                "card_metadata": card_meta,
            }
            for _, candidate, card_meta in candidates
        ],
    }
    if settings.diagnostics_enabled or settings.dry_run:
        with open(settings.diagnostics_file, "w", encoding="utf-8") as handle:
            json.dump(diagnostics, handle, indent=2, ensure_ascii=False)

    log(
        f"V2 discovered={len(articles)} selected={len(candidates)} "
        f"reasons={selection['reasons']}"
    )
    for _, candidate, _ in candidates:
        log(
            f"selected {build_post_text(candidate)} | {candidate.title} | "
            f"{candidate.post_url}"
        )

    if settings.dry_run:
        for _, candidate, _ in candidates:
            log(
                f"DRY_RUN would post text={build_post_text(candidate)!r} "
                f"card_title={candidate.title!r}"
            )
        # A dry run must never mutate production dedupe state.
        return

    if not candidates:
        save_state(settings.state_file, state)
        return
    if not settings.bsky_identifier or not settings.bsky_app_password:
        raise RuntimeError(
            "BSKY_IDENTIFIER and BSKY_APP_PASSWORD are required when not DRY_RUN"
        )

    session = requests.Session()
    did, jwt = bsky_login(
        session,
        settings.bsky_pds,
        settings.bsky_identifier,
        settings.bsky_app_password,
        settings.request_timeout,
    )
    for article, candidate, _ in candidates:
        post_to_bluesky(
            session,
            candidate,
            settings.bsky_pds,
            did,
            jwt,
            settings.request_timeout,
        )
        mark_posted(state, article)
        # Persist each successful post locally so an always-run workflow state
        # commit can preserve partial progress if a later post fails.
        save_state(settings.state_file, state)
        log(f"posted {candidate.post_url}")
        time.sleep(0.8)


if __name__ == "__main__":
    main()
