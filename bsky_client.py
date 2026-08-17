from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import requests
from PIL import Image, ImageOps, UnidentifiedImageError

from models import Candidate


IMAGE_MAX_BYTES = 2_000_000

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


def _source_text(candidate: Candidate) -> str:
    source = display_source_name(candidate.source)
    if candidate.access == "paywalled":
        source = f"{source} ($)"
    return source


def _source_author_text(candidate: Candidate) -> str:
    source = _source_text(candidate)
    author = display_author(candidate.author, candidate.source)
    return f"{source} · {author}" if author else source


def _article_url(candidate: Candidate) -> str:
    return candidate.post_url or candidate.canonical_url or candidate.publisher_url or candidate.url


def is_game_thread_candidate(candidate: Candidate) -> bool:
    return (candidate.discovered_via or "").startswith("game_thread:")


def _post_prefix(candidate: Candidate) -> str:
    source_author = _source_author_text(candidate)
    return f"Game recap · {source_author}" if is_game_thread_candidate(candidate) else source_author


def _headline_post_text(prefix: str, headline: str, max_len: int = 290) -> str:
    prefix = truncate_line(prefix, max_len)
    headline = (headline or "").strip()
    if not headline:
        return prefix
    remaining = max(1, max_len - len(prefix) - 1)
    return f"{prefix}\n{truncate_line(headline, remaining)}"


def build_game_post_text(candidate: Candidate) -> str:
    return _headline_post_text(_post_prefix(candidate), candidate.title)


def build_post_text(candidate: Candidate) -> str:
    return _headline_post_text(_post_prefix(candidate), candidate.title)


def build_link_post(candidate: Candidate) -> tuple[str, list[dict]]:
    """Build post text with a compact clickable publisher link on the final line."""
    link_label = f"Read at {display_source_name(candidate.source)} →"
    body_max = max(1, 290 - len(link_label) - 1)
    body = _headline_post_text(_post_prefix(candidate), candidate.title, body_max)
    text = f"{body}\n{link_label}"

    byte_start = len(f"{body}\n".encode("utf-8"))
    byte_end = byte_start + len(link_label.encode("utf-8"))
    facets = [{
        "index": {"byteStart": byte_start, "byteEnd": byte_end},
        "features": [{
            "$type": "app.bsky.richtext.facet#link",
            "uri": _article_url(candidate),
        }],
    }]
    return text, facets


def bsky_login(session: requests.Session, pds: str, identifier: str, app_password: str, timeout: int) -> Tuple[str, str]:
    r = session.post(
        f"{pds}/xrpc/com.atproto.server.createSession",
        json={"identifier": identifier, "password": app_password},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return data["did"], data["accessJwt"]


def _image_aspect_ratio(blob_bytes: bytes) -> Optional[Dict[str, int]]:
    """Return the displayed image ratio, respecting EXIF rotation when present."""
    try:
        with Image.open(io.BytesIO(blob_bytes)) as image:
            oriented = ImageOps.exif_transpose(image)
            try:
                width, height = oriented.size
            finally:
                if oriented is not image:
                    oriented.close()
    except (UnidentifiedImageError, OSError, ValueError):
        return None

    if width < 1 or height < 1:
        return None
    return {"width": int(width), "height": int(height)}


def upload_image_blob(
    session: requests.Session,
    image_url: str,
    pds: str,
    jwt: str,
    timeout: int,
) -> Optional[Tuple[Dict[str, Any], Dict[str, int]]]:
    if not image_url:
        return None
    try:
        with session.get(image_url, timeout=timeout, stream=True) as r:
            if r.status_code >= 400:
                return None
            content_type = r.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                return None
            # Read one byte beyond the lexicon limit so an oversize image is
            # rejected cleanly instead of being silently truncated.
            blob_bytes = r.raw.read(IMAGE_MAX_BYTES + 1)
            if not blob_bytes or len(blob_bytes) > IMAGE_MAX_BYTES:
                return None
    except Exception:
        return None

    aspect_ratio = _image_aspect_ratio(blob_bytes)
    if not aspect_ratio:
        return None

    up = session.post(
        f"{pds}/xrpc/com.atproto.repo.uploadBlob",
        headers={"Authorization": f"Bearer {jwt}", "Content-Type": content_type},
        data=blob_bytes,
        timeout=timeout,
    )
    up.raise_for_status()
    blob = up.json().get("blob")
    if not blob:
        return None
    return blob, aspect_ratio


def create_image_embed(
    candidate: Candidate,
    image_blob: Dict[str, Any],
    aspect_ratio: Dict[str, int],
) -> Dict[str, Any]:
    alt = (candidate.title or f"Image from {display_source_name(candidate.source)}").strip()
    return {
        "$type": "app.bsky.embed.images",
        "images": [{
            "image": image_blob,
            "alt": truncate_line(alt, 1000),
            "aspectRatio": aspect_ratio,
        }],
    }


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
    image_upload = upload_image_blob(session, candidate.image_url, pds, jwt, timeout)
    text, facets = build_link_post(candidate)

    record: Dict[str, Any] = {
        "$type": "app.bsky.feed.post",
        "text": text,
        "facets": facets,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if image_upload:
        image_blob, aspect_ratio = image_upload
        record["embed"] = create_image_embed(candidate, image_blob, aspect_ratio)

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
