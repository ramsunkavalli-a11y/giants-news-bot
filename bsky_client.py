from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import requests

from models import Candidate


DISPLAY_SOURCE_NAMES = {
    "SF Standard": "San Francisco Standard",
    "SFGate Giants": "SFGATE",
    "NYTimes Baseball": "The New York Times",
    "NBC Sports Bay Area": "NBC Sports Bay Area",
    "SF Chronicle Giants": "SF Chronicle",
    "San Francisco Chronicle": "SF Chronicle",
    "Mercury News Giants": "Mercury News",
    "AP Giants": "Associated Press",
    "AP Giants hub": "Associated Press",
    "MLB Giants": "MLB.com",
    "MLB Giants News": "MLB.com",
    "Fangraphs Giants": "FanGraphs",
    "Baseball America Giants": "Baseball America",
    "KNBR Giants": "KNBR",
}

OUTLET_AUTHOR_NAMES = {
    "san francisco standard",
    "sf standard",
    "sfgate",
    "the new york times",
    "new york times",
    "nbc sports bay area",
    "san francisco chronicle",
    "sf chronicle",
    "mercury news",
    "the mercury news",
    "associated press",
    "the associated press",
    "ap",
    "ap news",
    "mlb.com",
    "major league baseball",
    "fangraphs",
    "baseball america",
    "knbr",
}


def truncate_line(line: str, max_len: int) -> str:
    if len(line) <= max_len:
        return line
    return line[: max_len - 1].rstrip() + "…"


def display_source_name(source: str) -> str:
    return DISPLAY_SOURCE_NAMES.get(source, source or "Giants News")


def _looks_like_domain(value: str) -> bool:
    return bool(re.fullmatch(r"(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+", value))


def display_author(author: str, source: str) -> str:
    cleaned = re.sub(r"^\s*by\s+", "", (author or "").strip(), flags=re.I)
    if not cleaned:
        return ""
    normalized = re.sub(r"\s+", " ", cleaned).strip().lower()
    if normalized in OUTLET_AUTHOR_NAMES:
        return ""
    if _looks_like_domain(normalized):
        return ""
    if normalized == (source or "").strip().lower():
        return ""
    if normalized == display_source_name(source).lower():
        return ""
    return cleaned


def _source_author_text(candidate: Candidate) -> str:
    source = display_source_name(candidate.source)
    if candidate.access == "paywalled":
        source = f"{source} ($)"
    author = display_author(candidate.author, candidate.source)
    return f"{source} · {author}" if author else source


def is_game_thread_candidate(candidate: Candidate) -> bool:
    return (candidate.discovered_via or "").startswith("game_thread:")


def build_game_post_text(candidate: Candidate) -> str:
    prefix = f"Game recap · {_source_author_text(candidate)}"
    headline = (candidate.title or "").strip()
    if not headline:
        return truncate_line(prefix, 290)
    remaining = max(1, 290 - len(prefix) - 1)
    return f"{prefix}\n{truncate_line(headline, remaining)}"


def build_post_text(candidate: Candidate) -> str:
    if is_game_thread_candidate(candidate):
        return build_game_post_text(candidate)
    return truncate_line(_source_author_text(candidate), 290)


def bsky_login(session: requests.Session, pds: str, identifier: str, app_password: str, timeout: int) -> Tuple[str, str]:
    r = session.post(
        f"{pds}/xrpc/com.atproto.server.createSession",
        json={"identifier": identifier, "password": app_password},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return data["did"], data["accessJwt"]


def upload_external_thumb(session: requests.Session, image_url: str, pds: str, jwt: str, timeout: int) -> Optional[Dict[str, Any]]:
    if not image_url:
        return None
    try:
        with session.get(image_url, timeout=timeout, stream=True) as r:
            if r.status_code >= 400:
                return None
            content_type = r.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                return None
            blob_bytes = r.raw.read(900_000)
            if not blob_bytes:
                return None
    except Exception:
        return None
    up = session.post(
        f"{pds}/xrpc/com.atproto.repo.uploadBlob",
        headers={"Authorization": f"Bearer {jwt}", "Content-Type": content_type},
        data=blob_bytes,
        timeout=timeout,
    )
    up.raise_for_status()
    return up.json().get("blob")


def create_embed_for_candidate(session: requests.Session, candidate: Candidate, pds: str, jwt: str, timeout: int) -> Dict[str, Any]:
    hide_card_text = is_game_thread_candidate(candidate)
    description = "" if hide_card_text else truncate_line(candidate.summary or display_source_name(candidate.source), 280)
    title = "" if hide_card_text else truncate_line(candidate.title or "Giants update", 100)
    external: Dict[str, Any] = {
        "uri": candidate.post_url or candidate.canonical_url or candidate.publisher_url or candidate.url,
        "title": title,
        "description": description,
    }
    thumb_blob = upload_external_thumb(session, candidate.image_url, pds, jwt, timeout)
    if thumb_blob:
        external["thumb"] = thumb_blob
    return {"$type": "app.bsky.embed.external", "external": external}


def post_to_bluesky(
    session: requests.Session,
    candidate: Candidate,
    pds: str,
    did: str,
    jwt: str,
    timeout: int,
    *,
    reply_root: Optional[Dict[str, str]] = None,
    reply_parent: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    record: Dict[str, Any] = {
        "$type": "app.bsky.feed.post",
        "text": build_post_text(candidate),
        "embed": create_embed_for_candidate(session, candidate, pds, jwt, timeout),
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if reply_root and reply_parent:
        record["reply"] = {
            "root": {"uri": reply_root["uri"], "cid": reply_root["cid"]},
            "parent": {"uri": reply_parent["uri"], "cid": reply_parent["cid"]},
        }

    payload = {
        "repo": did,
        "collection": "app.bsky.feed.post",
        "record": record,
    }
    r = session.post(
        f"{pds}/xrpc/com.atproto.repo.createRecord",
        headers={"Authorization": f"Bearer {jwt}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return {"uri": data["uri"], "cid": data["cid"]}
