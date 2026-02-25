from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from config import Settings
from filters import canonicalize_url, parse_dt_or_none


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def state_load(settings: Settings) -> Dict[str, Any]:
    if not os.path.exists(settings.state_file):
        return {"posted_urls": {}, "redirect_cache": {}, "meta_cache": {}}
    with open(settings.state_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("posted_urls", {})
    data.setdefault("redirect_cache", {})
    data.setdefault("meta_cache", {})

    normalized_redirect_cache: Dict[str, str] = {}
    for raw_key, raw_val in data["redirect_cache"].items():
        key = canonicalize_url(raw_key, settings)
        if not key:
            continue
        val = canonicalize_url(raw_val.get("final", ""), settings) if isinstance(raw_val, dict) else canonicalize_url(raw_val, settings)
        normalized_redirect_cache[key] = val or key
    data["redirect_cache"] = normalized_redirect_cache
    return data


def state_save(settings: Settings, state: Dict[str, Any]) -> None:
    with open(settings.state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def prune_state(settings: Settings, state: Dict[str, Any]) -> None:
    posted_cutoff = now_utc() - timedelta(days=settings.keep_posted_days)
    for url, ts in list(state["posted_urls"].items()):
        dt = parse_dt_or_none(ts)
        if not dt or dt < posted_cutoff:
            del state["posted_urls"][url]

    meta_cutoff = now_utc() - timedelta(days=settings.meta_cache_days)
    for url, data in list(state["meta_cache"].items()):
        ts = data.get("ts", "") if isinstance(data, dict) else ""
        dt = parse_dt_or_none(ts)
        if not dt or dt < meta_cutoff:
            del state["meta_cache"][url]
