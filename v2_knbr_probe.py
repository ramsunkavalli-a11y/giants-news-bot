from __future__ import annotations

import json

from v2_knbr import discover_knbr_executive_show


def main() -> None:
    articles = discover_knbr_executive_show()
    if not articles:
        raise RuntimeError("KNBR Executive Show probe returned no Giants episodes")
    payload = {
        "count": len(articles),
        "latest": [
            {
                "title": article.title,
                "url": article.url,
                "published": article.published,
                "author": article.author,
            }
            for article in articles[:8]
        ],
    }
    with open("v2-knbr-probe.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"KNBR_EXECUTIVE_SHOW count={len(articles)}")
    for article in articles[:5]:
        print(f"  {article.published} | {article.title} | {article.url}")


if __name__ == "__main__":
    main()
