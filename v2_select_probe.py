from __future__ import annotations

import argparse
import json

from v2_selector import select_articles


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default="v2-probe.json")
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", default="v2-selection.json")
    args = parser.parse_args()

    probe = load_json(args.probe)
    state = load_json(args.state)
    result = select_articles(probe.get("articles", []), state)

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print(
        f"SELECTION_SIM state_urls={result['production_posted_url_count']} "
        f"history_stories={result['recent_posted_story_count']} "
        f"cutoff={result['cutoff']} selected={len(result['selected'])}"
    )
    for index, article in enumerate(result["selected"], start=1):
        byline = f" · {article.get('author')}" if article.get("author") else ""
        access = " ($)" if article.get("access") == "paywalled" else ""
        print(
            f"WOULD_POST {index}: {article.get('source')}{access}{byline} | "
            f"{article.get('title')} | {article.get('published')} | {article.get('url')}"
        )
    print("REASONS " + json.dumps(result["reasons"], sort_keys=True))

    duplicate_clusters = [item for item in result["clusters"] if item["member_count"] > 1]
    if duplicate_clusters:
        print("DUPLICATE_CLUSTERS:")
        for item in duplicate_clusters:
            print(
                f"  CHOSE {item['chosen_source']} · {item['chosen_author']} | "
                f"{item['chosen_title']}"
            )
            for alternative in item["alternatives"]:
                print(
                    f"    OVER {alternative['source']} · {alternative['author']} | "
                    f"{alternative['title']}"
                )


if __name__ == "__main__":
    main()
