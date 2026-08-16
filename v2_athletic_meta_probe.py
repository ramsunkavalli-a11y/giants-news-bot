from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import feedparser
import requests
from bs4 import BeautifulSoup

UA = "Mozilla/5.0 GiantsNewsBotV2Probe/0.4"
TIMEOUT = 20
FEED_URL = "https://www.nytimes.com/athletic/rss/mlb/sf-giants/"


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def author_names(value) -> list[str]:
    names: list[str] = []
    if isinstance(value, str):
        if clean(value):
            names.append(clean(value))
    elif isinstance(value, dict):
        name = clean(str(value.get("name", "")))
        if name:
            names.append(name)
    elif isinstance(value, list):
        for item in value:
            names.extend(author_names(item))
    return names


def jsonld_authors(soup: BeautifulSoup) -> tuple[list[str], list[str]]:
    authors: list[str] = []
    types: list[str] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text()
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
                continue
            if not isinstance(node, dict):
                continue
            node_type = node.get("@type")
            if isinstance(node_type, list):
                types.extend(clean(str(t)) for t in node_type if clean(str(t)))
            elif node_type:
                types.append(clean(str(node_type)))
            if "author" in node:
                authors.extend(author_names(node["author"]))
            for key in ("@graph", "mainEntity"):
                child = node.get(key)
                if isinstance(child, (dict, list)):
                    stack.append(child)
    return list(dict.fromkeys(authors)), list(dict.fromkeys(types))


def meta_values(soup: BeautifulSoup, keys: tuple[str, ...]) -> dict[str, str]:
    found: dict[str, str] = {}
    for key in keys:
        tag = soup.find("meta", attrs={"name": key}) or soup.find("meta", attrs={"property": key})
        if tag and clean(tag.get("content", "")):
            found[key] = clean(tag.get("content", ""))
    return found


def main() -> None:
    feed = feedparser.parse(FEED_URL, request_headers={"User-Agent": UA})
    rows = []
    for entry in feed.entries[:8]:
        url = getattr(entry, "link", "") or ""
        title = clean(getattr(entry, "title", ""))
        row = {"title": title, "url": url}
        try:
            response = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT, allow_redirects=True)
            row.update({
                "status": response.status_code,
                "final_url": response.url,
                "length": len(response.text),
                "content_type": response.headers.get("content-type", ""),
            })
            soup = BeautifulSoup(response.text, "lxml")
            authors, types = jsonld_authors(soup)
            row["jsonld_authors"] = authors
            row["jsonld_types"] = types[:20]
            row["meta"] = meta_values(
                soup,
                (
                    "author",
                    "byl",
                    "parsely-author",
                    "article:author",
                    "og:site_name",
                ),
            )
            row["title_tag"] = clean(soup.title.get_text(" ")) if soup.title else ""
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "rows": rows}
    with open("v2-athletic-meta-probe.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    for row in rows:
        print(
            "ATH_META "
            f"status={row.get('status')} length={row.get('length')} "
            f"authors={row.get('jsonld_authors', [])} meta={row.get('meta', {})} "
            f"title={row['title']}"
        )


if __name__ == "__main__":
    main()
