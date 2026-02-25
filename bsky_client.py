from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from models import Candidate


def truncate_line(line: str, max_len: int) -> str:
    if len(line) <= max_len:
        return line
    return line[: max_len - 1].rstrip() + "…"


def build_post_text(candidate: Candidate) -> str:
    first = f"{candidate.source}: {candidate.title or 'Giants update'}"
    first = truncate_line(first, 260)
    return f"{first}\n{candidate.url}"


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
    description = truncate_line(candidate.summary or candidate.source, 280)
    external: Dict[str, Any] = {
        "uri": candidate.url,
        "title": truncate_line(candidate.title or "Giants update", 100),
        "description": description,
    }
    thumb_blob = upload_external_thumb(session, candidate.image_url, pds, jwt, timeout)
    if thumb_blob:
        external["thumb"] = thumb_blob
    return {"$type": "app.bsky.embed.external", "external": external}


def post_to_bluesky(session: requests.Session, candidate: Candidate, pds: str, did: str, jwt: str, timeout: int) -> None:
    text = build_post_text(candidate)
    link_start = text.rfind(candidate.url)
    facets: List[Dict[str, Any]] = []
    if link_start >= 0:
        start_bytes = len(text[:link_start].encode("utf-8"))
        end_bytes = start_bytes + len(candidate.url.encode("utf-8"))
        facets.append(
            {
                "index": {"byteStart": start_bytes, "byteEnd": end_bytes},
                "features": [{"$type": "app.bsky.richtext.facet#link", "uri": candidate.url}],
            }
        )

    payload = {
        "repo": did,
        "collection": "app.bsky.feed.post",
        "record": {
            "$type": "app.bsky.feed.post",
            "text": text,
            "facets": facets,
            "embed": create_embed_for_candidate(session, candidate, pds, jwt, timeout),
            "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    }
    r = session.post(
        f"{pds}/xrpc/com.atproto.repo.createRecord",
        headers={"Authorization": f"Bearer {jwt}"},
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
