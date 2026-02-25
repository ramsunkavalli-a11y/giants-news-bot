# giants-news-bot

Python GitHub-Actions-friendly bot that discovers SF Giants articles, scores candidates, and posts top links to Bluesky.

## Architecture

- `config.py`: env vars + source configuration (`rss_only`, `google_news`, `non_rss`).
- `models.py`: candidate dataclass with scoring + diagnostics fields.
- `discovery_rss.py`: RSS-only source discovery and feed health metrics.
- `discovery_google.py`: Google News RSS discovery.
- `discovery_nonrss.py`: bounded sitemap/listing discovery for non-RSS sources.
- `parser_meta.py`: BeautifulSoup/lxml metadata + JSON-LD extraction.
- `filters.py`: canonicalization, story URL checks, Giants relevance checks.
- `scoring.py`: transparent scoring model (author priority is boost only).
- `state.py`: state load/save/prune logic.
- `bsky_client.py`: ATProto login, external embed, posting.
- `main.py`: orchestration, validation, selection, logging, diagnostics artifact.

## Behavior highlights

- RSS-only sources (`SF Standard`, `SFGate Giants`, `NYTimes Baseball`) use RSS discovery only.
- Author priority no longer hard-rejects candidates; it contributes score boost.
- RSS health logs include feed URL, bozo, status, entry count.
- Per-source rejection reasons are logged and captured in diagnostics JSON.

## Running

```bash
pip install -r requirements.txt
DRY_RUN=1 python giants_news_bot.py
```

## Diagnostics JSON

`diagnostics.json` is written when:
- `DRY_RUN=1`, or
- `DIAGNOSTICS_ENABLED=1`

Path can be overridden via `DIAGNOSTICS_FILE`.

## GitHub Actions artifact example

```yaml
- name: Run bot
  env:
    DRY_RUN: '1'
    DIAGNOSTICS_ENABLED: '1'
  run: python giants_news_bot.py

- name: Upload diagnostics
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: diagnostics
    path: diagnostics.json
```
