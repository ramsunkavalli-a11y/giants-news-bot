from __future__ import annotations

import requests

import main as bot_main
from bsky_client import display_author
from config import Settings
from parser_meta import extract_meta


_original_cand_dict = bot_main._cand_dict


def _audit_cand_dict(candidate):
    payload = _original_cand_dict(candidate)
    payload.update(
        {
            "author": candidate.author,
            "summary": candidate.summary,
            "image_url": candidate.image_url,
            "published_ts": candidate.published_ts,
            "discovered_via": candidate.discovered_via,
            "categories": candidate.categories,
            "meta_sources_used": candidate.meta_sources_used,
        }
    )
    return payload


def _run_smoke_checks() -> None:
    assert display_author("sfchronicle.com", "SF Chronicle Giants") == ""
    assert display_author("AP News", "AP Giants") == ""

    url = "https://www.nbcsportsbayarea.com/mlb/san-francisco-giants/2028-all-star-game-oracle-park/1956590/"
    settings = Settings()
    response = requests.get(url, headers={"User-Agent": settings.user_agent}, timeout=settings.request_timeout)
    response.raise_for_status()
    meta = extract_meta(response.url, response.text)
    assert meta.image_url.startswith("http"), f"unexpected NBC image metadata: {meta.image_url!r}"
    print(f"NBC metadata smoke image={meta.image_url}")


bot_main._cand_dict = _audit_cand_dict


if __name__ == "__main__":
    _run_smoke_checks()
    bot_main.main()
