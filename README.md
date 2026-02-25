# giants-news-bot

Python GitHub-Actions-friendly SF Giants → Bluesky bot.

## Key behaviors

- RSS-only sources: SF Standard, SFGate Giants, NYTimes Baseball.
- Non-RSS crawling excludes MLB.com listing/sitemap discovery.
- Strict MLB article URL validation (`/giants/news/` + article-like slug + blocked-path exclusions).
- Author priority is a score boost, not a hard gate.
- Request/runtime guardrails:
  - `MAX_TOTAL_HTTP_REQUESTS`
  - `MAX_DISCOVERY_SECONDS`
  - `MAX_ENRICH_CANDIDATES`
- DRY_RUN writes `diagnostics.json` with source-level rejection reasons.

## Run

```bash
pip install -r requirements.txt
DRY_RUN=1 python giants_news_bot.py
```
