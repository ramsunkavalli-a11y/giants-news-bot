# giants-news-bot

Python GitHub-Actions-friendly SF Giants → Bluesky bot.

## Key behaviors

- RSS-only sources: SF Standard, SFGate Giants, NYTimes Baseball.
- Google News wrappers are canonicalized to publisher URLs before scoring/dedupe/posting.
- Non-RSS crawling excludes MLB.com listing/sitemap discovery.
- Strict MLB article URL validation (`/giants/news/` + article-like slug + blocked-path exclusions).
- Source policies filter noisy/non-article pages for FanGraphs, Baseball America, AP hubs, and generic Bay Area sports pages.
- Author priority is a score boost, not a hard gate.
- Request/runtime guardrails:
  - `MAX_TOTAL_HTTP_REQUESTS`
  - `MAX_DISCOVERY_SECONDS`
  - `MAX_ENRICH_CANDIDATES`
- DRY_RUN writes `diagnostics.json` with stage-level outcomes and skip reasons.

## Run

```bash
pip install -r requirements.txt
DRY_RUN=1 DIAGNOSTICS_ENABLED=1 python giants_news_bot.py
```
