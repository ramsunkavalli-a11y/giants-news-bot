from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup

from bsky_client import bsky_login, build_post_text, post_to_bluesky
from config import Settings
from models import Candidate
from v2_game_threads import is_game_story, select_game_threads
from v2_probe import (
    discover_athletic,
    discover_fangraphs,
    discover_mlb,
    discover_nbc,
    discover_sf_standard,
    discover_sfgate,
)
from v2_radar import discover_core_writer_radar
from v2_selector import canonicalize_url, parse_dt, select_articles

DISCOVERERS = [
    discover_sf_standard,
    discover_athletic,
    discover_mlb,
    discover_sfgate,
    discover_fangraphs,
    discover_nbc,
    discover_core_writer_radar,
]

PROMO_SUMMARY_PATTERNS = (
    "this story was excerpted from",
    "to read the full newsletter",
    "subscribe to get it regularly",
    "subscribe to our newsletter",
)


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {
            "posted_urls": {},
            "posted_stories": [],
            "game_threads": {},
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
    if not isinstance(state.get("game_threads"), dict):
        state["game_threads"] = {}
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

    threads = state.get("game_threads", {})
    if isinstance(threads, dict):
        for key, item in list(threads.items()):
            if not isinstance(item, dict):
                threads.pop(key, None)
                continue
            dt = parse_dt(item.get("updated_at", "") or item.get("created_at", ""))
            if dt is not None and dt < cutoff:
                threads.pop(key, None)


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


def clean_card_summary(*values: str) -> str:
    for value in values:
        if not value:
            continue
        text = BeautifulSoup(value, "html.parser").get_text(" ")
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        lower = text.lower()
        if any(pattern in lower for pattern in PROMO_SUMMARY_PATTERNS):
            continue
        return text
    return ""


def article_to_candidate(article: dict, card_meta: dict) -> Candidate:
    url = (
        article.get("canonical_url")
        or canonicalize_url(article.get("url", ""))
        or article.get("url", "")
    )
    summary = clean_card_summary(
        article.get("summary", ""),
        card_meta.get("description", ""),
    )
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


def mark_posted(
    state: dict,
    article: dict,
    *,
    kind: str = "standalone",
    game_key: str = "",
) -> None:
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
        "kind": kind,
        "game_key": game_key,
        "posted_at": ts,
    })


def _valid_ref(value) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("uri"), str)
        and bool(value.get("uri"))
        and isinstance(value.get("cid"), str)
        and bool(value.get("cid"))
    )


def _existing_thread_key(state: dict, thread: dict) -> str:
    threads = state.get("game_threads", {})
    key = thread.get("key", "")
    if key in threads:
        return key

    day = thread.get("game_day", "")
    unknown = f"game:{day}:unknown"
    if unknown in threads:
        return unknown

    day_prefix = f"game:{day}:"
    same_day = [candidate for candidate in threads if candidate.startswith(day_prefix)]
    if len(same_day) == 1:
        return same_day[0]
    return key


def _set_thread_state(
    state: dict,
    key: str,
    thread: dict,
    root: dict,
    parent: dict,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    threads = state.setdefault("game_threads", {})
    existing = threads.get(key, {}) if isinstance(threads.get(key), dict) else {}
    threads[key] = {
        "game_day": thread.get("game_day", ""),
        "opponent": thread.get("opponent", "") or existing.get("opponent", ""),
        "root": root,
        "parent": parent,
        "created_at": existing.get("created_at", now),
        "updated_at": now,
    }


def _prepare_posts(articles: list[dict], timeout: int) -> list[tuple[dict, Candidate, dict]]:
    prepared = []
    for article in articles:
        card_meta = enrich_card_metadata(article.get("url", ""), timeout)
        prepared.append((article, article_to_candidate(article, card_meta), card_meta))
    return prepared


def main() -> None:
    settings = Settings()
    state = load_state(settings.state_file)
    prune_state(state, settings.keep_posted_days)

    articles, health = discover_articles()
    game_hours_back = int(os.getenv("GAME_HOURS_BACK", "36"))

    game_selection = select_game_threads(
        articles,
        state,
        hours_back=game_hours_back,
    )
    standalone_articles = [article for article in articles if not is_game_story(article)]
    selection = select_articles(
        standalone_articles,
        state,
        hours_back=settings.hours_back,
        max_posts=settings.max_posts_per_run,
    )

    candidates = _prepare_posts(selection["selected"], settings.request_timeout)
    game_candidates = []
    for thread in game_selection["threads"]:
        game_candidates.append({
            **{key: value for key, value in thread.items() if key != "articles"},
            "posts": _prepare_posts(thread["articles"], settings.request_timeout),
        })

    diagnostics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": settings.dry_run,
        "health": health,
        "selection": selection,
        "game_selection": game_selection,
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
        "game_threads": [
            {
                "key": thread["key"],
                "game_day": thread["game_day"],
                "opponent": thread["opponent"],
                "existing_thread_key": _existing_thread_key(state, thread),
                "posts": [
                    {
                        "text": build_post_text(candidate),
                        "title": candidate.title,
                        "url": candidate.post_url,
                        "source": candidate.source,
                        "author": candidate.author,
                        "access": candidate.access,
                    }
                    for _, candidate, _ in thread["posts"]
                ],
            }
            for thread in game_candidates
        ],
    }
    if settings.diagnostics_enabled or settings.dry_run:
        with open(settings.diagnostics_file, "w", encoding="utf-8") as handle:
            json.dump(diagnostics, handle, indent=2, ensure_ascii=False)

    game_story_count = sum(len(thread["posts"]) for thread in game_candidates)
    log(
        f"V2 discovered={len(articles)} standalone={len(candidates)} "
        f"game_threads={len(game_candidates)} game_stories={game_story_count} "
        f"reasons={selection['reasons']} game_reasons={game_selection['reasons']}"
    )
    for _, candidate, _ in candidates:
        log(
            f"selected standalone {build_post_text(candidate)} | {candidate.title} | "
            f"{candidate.post_url}"
        )
    for thread in game_candidates:
        existing_key = _existing_thread_key(state, thread)
        existing = state.get("game_threads", {}).get(existing_key, {})
        mode = "append" if _valid_ref(existing.get("root")) else "start"
        log(
            f"selected game_thread key={thread['key']} mode={mode} "
            f"stories={len(thread['posts'])} opponent={thread['opponent'] or 'unknown'}"
        )

    if settings.dry_run:
        for _, candidate, _ in candidates:
            log(
                f"DRY_RUN would post standalone text={build_post_text(candidate)!r} "
                f"card_title={candidate.title!r}"
            )
        for thread in game_candidates:
            existing_key = _existing_thread_key(state, thread)
            existing = state.get("game_threads", {}).get(existing_key, {})
            has_root = _valid_ref(existing.get("root"))
            for index, (_, candidate, _) in enumerate(thread["posts"]):
                action = "reply" if has_root or index > 0 else "root"
                log(
                    f"DRY_RUN game_thread={existing_key or thread['key']} action={action} "
                    f"text={build_post_text(candidate)!r} card_title={candidate.title!r}"
                )
        return

    if not candidates and not game_candidates:
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
        save_state(settings.state_file, state)
        log(f"posted standalone {candidate.post_url}")
        time.sleep(0.8)

    for thread in game_candidates:
        state_key = _existing_thread_key(state, thread)
        existing = state.get("game_threads", {}).get(state_key, {})
        root = existing.get("root") if _valid_ref(existing.get("root")) else None
        parent = existing.get("parent") if _valid_ref(existing.get("parent")) else root

        for article, candidate, _ in thread["posts"]:
            if root:
                ref = post_to_bluesky(
                    session,
                    candidate,
                    settings.bsky_pds,
                    did,
                    jwt,
                    settings.request_timeout,
                    reply_root=root,
                    reply_parent=parent or root,
                )
            else:
                ref = post_to_bluesky(
                    session,
                    candidate,
                    settings.bsky_pds,
                    did,
                    jwt,
                    settings.request_timeout,
                )
                root = ref
            parent = ref
            _set_thread_state(state, state_key, thread, root, parent)
            mark_posted(state, article, kind="game_story", game_key=state_key)
            save_state(settings.state_file, state)
            log(f"posted game_thread={state_key} {candidate.post_url}")
            time.sleep(0.8)


if __name__ == "__main__":
    main()
