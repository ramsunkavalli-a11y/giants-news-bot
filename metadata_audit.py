from __future__ import annotations

import main as bot_main


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


bot_main._cand_dict = _audit_cand_dict


if __name__ == "__main__":
    bot_main.main()

# audit rerun: image + pseudo-author cleanup
