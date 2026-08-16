from __future__ import annotations

import json
from urllib.parse import urlencode, urlparse

import feedparser

try:
    from googlenewsdecoder import gnewsdecoder
except Exception:
    gnewsdecoder = None

UA = "Mozilla/5.0 GiantsNewsBotV2RadarProbe/0.1"

QUERIES = {
    "broad": '"San Francisco Giants" when:72h',
    "slusser": 'site:sfchronicle.com "Susan Slusser" Giants when:168h',
    "rubin": 'site:sfchronicle.com "Shayna Rubin" Giants when:168h',
    "delos_santos": 'site:mercurynews.com "Justice delos Santos" Giants when:168h',
}


def google_rss(query: str) -> str:
    params = urlencode({"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    return f"https://news.google.com/rss/search?{params}"


def decode(url: str) -> dict:
    if not gnewsdecoder:
        return {"ok": False, "url": "", "reason": "decoder_unavailable"}
    try:
        value = gnewsdecoder(url)
        decoded = str(value.get("decoded_url") or "") if isinstance(value, dict) else ""
        return {
            "ok": bool(isinstance(value, dict) and value.get("status") and decoded),
            "url": decoded,
            "domain": urlparse(decoded).netloc.lower() if decoded else "",
            "raw": value,
        }
    except Exception as exc:
        return {"ok": False, "url": "", "reason": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    payload = {}
    for name, query in QUERIES.items():
        url = google_rss(query)
        feed = feedparser.parse(url, request_headers={"User-Agent": UA})
        entries = []
        for entry in feed.entries[:10]:
            source = getattr(entry, "source", {})
            source_title = source.get("title", "") if isinstance(source, dict) else ""
            link = getattr(entry, "link", "") or ""
            entries.append({
                "title": getattr(entry, "title", "") or "",
                "published": getattr(entry, "published", "") or getattr(entry, "updated", "") or "",
                "source": source_title,
                "author": getattr(entry, "author", "") or "",
                "link": link,
                "decoded": decode(link),
                "keys": sorted(list(entry.keys())),
            })
        payload[name] = {
            "query": query,
            "status": getattr(feed, "status", None),
            "bozo": bool(getattr(feed, "bozo", False)),
            "count": len(feed.entries),
            "entries": entries,
        }

    with open("v2-radar-probe.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    for name, result in payload.items():
        print(f"RADAR {name} status={result['status']} count={result['count']} query={result['query']}")
        for item in result["entries"]:
            print(
                f"  {item['published']} | {item['source']} | author={item['author']!r} | "
                f"decoded={item['decoded'].get('domain', '')} ok={item['decoded'].get('ok')} | {item['title']}"
            )


if __name__ == "__main__":
    main()
