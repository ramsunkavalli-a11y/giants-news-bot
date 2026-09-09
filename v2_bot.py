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
from v2_knbr import discover_knbr_executive_show
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
    discover_knbr_executive_show,
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
            "run_history": [],
        }
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read().strip()
        state = json.loads(raw) if raw else {}
    except (OSError, json.JSONDecodeError):
        state = {}

    if not isinstance(state, dict):
        state = {}

    posted_urls = state.get("posted_urls", {})
    posted_stories = state.get("posted_stories", [])
    game_threads = state.get("game_threads", {})
    run_history = state.get("run_history", [])

    return {
        "posted_urls": posted_urls if isinstance(posted_urls, dict) else {},
        "posted_stories": posted_stories if isinstance(posted_stories, list) else [],
        "game_threads": game_threads if isinstance(game_threads, dict) else {},
        "run_history": run_history if isinstance(run_history, list) else [],
    }


def save_state(path: str, state: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True, ensure_ascii=False)


def record_run(
    state: dict,
    *,
    started_at: str,
    finished_at: str,
    status: str,
    health: dict,
    selection: dict,
    game_selection: dict,
    error: str = "",
) -> None:
    """Persist a compact production heartbeat alongside posting state.

    Diagnostics artifacts are useful for a single run, but they expire and are
    awkward to inspect without an authenticated Actions session. A bounded
    history in state.json makes a silent source outage, zero-selection run, or
    posting failure visible in the repository itself.
    """
    history = state.setdefault("run_history", [])
    if not isinstance(history, list):
        history = []
        state["run_history"] = history

    record = {
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
        "discovered": sum(
            int(item.get("count", 0) or 0)
            for item in health.values()
            if isinstance(item, dict) and item.get("ok")
        ),
        "sources_ok": sum(
            1 for item in health.values()
            if isinstance(item, dict) and item.get("ok")
        ),
        "sources_failed": sum(
            1 for item in health.values()
            if isinstance(item, dict) and not item.get("ok")
        ),
        "standalone_selected": len(selection.get("selected", []) or []),
        "game_threads_selected": len(game_selection.get("threads", []) or []),
        "game_stories_selected": sum(
            len(thread.get("articles", []) or [])
            for thread in game_selection.get("threads", []) or []
        ),
        "schedule_grounded_threads": sum(
            1
            for thread in game_selection.get("threads", []) or []
            if thread.get("schedule_grounded")
        ),
        "selection_reasons": selection.get("reasons", {}),
        "game_selection_reasons": game_selection.get("reasons", {}),
    }
    if error:
        record["error"] = error[:500]
    history.append(record)
    state["run_history"] = history[-100:]


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

    history = state.get("run_history", [])
    if isinstance(history, list):
        state["run_history"] = [
            item for item in history
            if isinstance(item, dict)
            and (
                (dt := parse_dt(item.get("finished_at", "") or item.get("timestamp", "")))
                is None
                or dt >= cutoff
            )
        ][-100:]


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

    # A schedule-backed game has a stable MLB identifier.  Never attach it to
    # a date-only/unknown legacy thread: that fallback can represent a
    # different game and would join unrelated recaps in one conversation.
    if thread.get("game_pk"):
        return key

    day = thread.get("game_day", "")
    opponent = thread.get("opponent", "")
    legacy_known = f"game:{day}:{opponent}" if day and opponent else ""
    if legacy_known and legacy_known in threads:
        return legacy_known

    # Date-only legacy grouping is retained only for existing unscheduled
    # threads. Never guess when a doubleheader or another ambiguity leaves
    # multiple matches.
    same_game = [
        candidate_key
        for candidate_key, candidate in threads.items()
        if isinstance(candidate, dict)
        and candidate.get("game_day") == day
        and (not opponent or candidate.get("opponent") == opponent)
    ]
    if len(same_game) == 1:
        return same_game[0]

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
        "game_pk": thread.get("game_pk", 0) or existing.get("game_pk", 0),
        "game_number": thread.get("game_number", 0) or existing.get("game_number", 0),
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


def _state_with_planned_game_stories(state: dict, game_selection: dict, now: datetime) -> dict:
    """Prevent a standalone from repeating a game-thread story in the same run."""
    planned = list(state.get("posted_stories", []) or [])
    for thread in game_selection.get("threads", []) or []:
        for article in thread.get("articles", []) or []:
            planned.append({
                "title": article.get("title", ""),
                "url": article.get("canonical_url") or canonicalize_url(article.get("url", "")),
                "source": article.get("source", ""),
                "author": article.get("author", ""),
                "kind": "game_story",
                "game_key": thread.get("key", ""),
                "posted_at": now.isoformat(),
            })
    return {**state, "posted_stories": planned}


def main() -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    settings = Settings()
    state = load_state(settings.state_file)
    prune_state(state, settings.keep_posted_days)

    articles, health = discover_articles()
    game_hours_back = int(os.getenv("GAME_HOURS_BACK", "30"))

    game_selection = select_game_threads(
        articles,
        state,
        hours_back=game_hours_back,
    )
    unrouted_game_articles = game_selection.get("unrouted_game_articles", []) or []
    standalone_articles = [article for article in articles if not is_game_story(article)]
    # A recap that cannot be tied to one scheduled game is still useful news;
    # it simply must not be threaded under an inferred game identity.
    standalone_articles.extend(unrouted_game_articles)
    selection = select_articles(
        standalone_articles,
        _state_with_planned_game_stories(state, game_selection, datetime.now(timezone.utc)),
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
                "game_pk": thread.get("game_pk", 0),
                "game_number": thread.get("game_number", 0),
                "game_day": thread["game_day"],
                "opponent": thread["opponent"],
                "schedule_grounded": thread.get("schedule_grounded", False),
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
            f"stories={len(thread['posts'])} opponent={thread['opponent'] or 'unknown'} "
            f"game_pk={thread.get('game_pk', 0)} grounded={thread.get('schedule_grounded', False)}"
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
        record_run(
            state,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            status="completed",
            health=health,
            selection=selection,
            game_selection=game_selection,
        )
        save_state(settings.state_file, state)
        return
    try:
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

        record_run(
            state,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            status="completed",
            health=health,
            selection=selection,
            game_selection=game_selection,
        )
        save_state(settings.state_file, state)
    except Exception as exc:
        record_run(
            state,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            status="failed",
            health=health,
            selection=selection,
            game_selection=game_selection,
            error=f"{type(exc).__name__}: {exc}",
        )
        save_state(settings.state_file, state)
        raise


if __name__ == "__main__":
    main()
