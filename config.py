from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    fallback = "1" if default else "0"
    return os.getenv(name, fallback).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Environment-backed settings used by the V2 production bot."""

    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "15"))
    dry_run: bool = _env_bool("DRY_RUN")
    state_file: str = os.getenv("STATE_FILE", "state.json")
    diagnostics_enabled: bool = _env_bool("DIAGNOSTICS_ENABLED")
    diagnostics_file: str = os.getenv("DIAGNOSTICS_FILE", "diagnostics.json")
    max_posts_per_run: int = int(os.getenv("MAX_POSTS_PER_RUN", "3"))
    hours_back: int = int(os.getenv("HOURS_BACK", "72"))
    keep_posted_days: int = int(os.getenv("KEEP_POSTED_DAYS", "21"))
    bsky_pds: str = os.getenv("BSKY_PDS", "https://bsky.social")
    bsky_identifier: str = os.getenv("BSKY_IDENTIFIER", "")
    bsky_app_password: str = os.getenv("BSKY_APP_PASSWORD", "")
