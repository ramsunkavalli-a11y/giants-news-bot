# Project context / new-session handoff

This is the quickest way for a new maintainer or a new ChatGPT conversation to understand the SF Giants News Bot without reconstructing prior discussions.

## What the product is

A Bluesky bot that posts a curated stream of San Francisco Giants journalism from multiple publications. The goal is **useful coverage, not maximum volume**. It should feel closer to an automatically maintained Giants news desk than an indiscriminate RSS firehose.

The account favors original reporting, breaking news, transactions, injuries, prospect work, meaningful analysis/features, and trusted beat reporting. It should avoid generic summaries, commodity recaps, promo pages, highlights/video-only pages, recurring evergreen content, and derivative articles whose main value is repeating another outlet.

Cost should remain essentially zero while the account is small. Prefer mature public feeds/parsers and deterministic logic over paid APIs, embeddings, or custom crawling infrastructure.

## Current production behavior

Production runs `v2_bot.py` from `.github/workflows/giants-news-bot.yml`.

### Cadence

Pacific local time:

- **Monday–Friday:** 8:30 AM, 2:30 PM, 7:30 PM, 11:30 PM
- **Saturday–Sunday:** 8:30 AM, 1:30 PM, 5:30 PM, 10:30 PM

The weekend shifts earlier because Giants weekends contain more day games and the first postgame stories often arrive in the afternoon. Four checks/day was chosen to reduce story delay and batch size without making the account feel continuously noisy.

GitHub schedules in UTC. The workflow includes both PDT and PST equivalents and an `America/Los_Angeles` gate so local times do not drift when daylight saving changes.

### Volume/freshness

- Standalone stories: maximum **3 per run**.
- Standalone discovery window: **72 hours**.
- Game-story window: **30 hours**.
- Game-thread replies are separate from the standalone cap.
- A run can legitimately post nothing.
- GitHub Actions scheduled jobs may start several minutes late; judge failures by whether the workflow actually ran, not by the exact wall-clock minute.

## Source philosophy

**Outsource/use structured discovery; own the last mile.** Do not build another universal crawler unless a compelling source leaves no cleaner option.

### Active direct sources

1. **SF Standard** — official Sports RSS
2. **The Athletic** — Giants RSS
3. **MLB.com** — Giants RSS
4. **SFGATE** — Giants RSS
5. **FanGraphs** — Giants category RSS
6. **NBC Sports Bay Area** — dedicated Giants news and analysis pages

These are implemented in `v2_probe.py`. Despite the filename, those discoverers are production code.

### Blocked-source radar

SF Chronicle and Mercury News are important, but their pages/feeds are unreliable from GitHub runners. Production therefore has a deliberately narrow radar in `v2_radar.py` for:

- Susan Slusser — SF Chronicle
- Shayna Rubin — SF Chronicle
- Justice delos Santos — Mercury News

For each writer the bot queries Google News RSS with the exact author + exact publisher domain + Giants, decodes the Google wrapper URL, verifies the resulting publisher domain, limits the results, rejects ambiguous multi-author-query attribution, and uses visible page metadata as an additional veto when available.

This is intentionally **not** broad Google News discovery. Broad Google results are noisy and are allowed only in diagnostics/probes. Do not quietly turn Google News into a general production feed.

### Trusted but inactive radar names

Baseball America, Associated Press, and KNBR remain publications worth considering in a future targeted adapter/radar, but they are **not currently active production discoverers**. Do not document or treat them as live until an adapter is actually added.

## Editorial priors

Author preference is useful for choosing among otherwise overlapping coverage; it is not a blanket permission to publish low-value material.

Current local/beat priors in `v2_authors.py`:

- **Elite:** Andrew Baggarly, Alex Pavlovic, Shayna Rubin
- **Very good:** Susan Slusser, Justice delos Santos
- **Good:** John Shea, Maria Guardado, Alex Simon; Kerry Crowley and Tim Kawakami when Giants-specific
- **Fine:** Grant Brisbee, Evan Webeck
- **National / high-value when Giants-specific:** Jeff Passan, Buster Olney, Jon Heyman, Bob Nightengale, Jon Morosi, Robert Murray, Ken Rosenthal, Evan Drellich

SFGATE has a mild `secondary` publication prior because its packaging can be click-driven. This should only be a light tie-breaker, not a hard suppression rule.

## Story/event dedupe

The system intentionally avoids a global numerical relevance score. Selection is mostly deterministic.

`v2_story.py` normalizes titles, recognizes event anchors/synonyms, and clusters stories that describe the same event within a time window. Examples that should cluster include multiple reports of the same surgery, retirement, trade, signing, promotion, or hosting announcement. Two different analyses about the same player should not cluster merely because the player name matches.

Within a duplicate cluster, author preference is the strongest tie-breaker, followed by light source preference, a real named byline, and recency.

Important rule: **choose the best article in an event before applying the one-source-per-run rule.** If the best article's publication has already been used for that standalone run, skip the event rather than falling through to a weaker duplicate just to fill the quota.

Historical dedupe also compares against recent `posted_stories` and canonical posted URLs in `state.json`.

## Standalone selection

Standalone news is for high-value material that is not being handled as game coverage.

- High-quality candidates only.
- Maximum 3/run.
- One standalone story per publication per run.
- Cross-publisher story/event duplicates collapsed to one winner.
- Missing timestamps are enriched where possible; unresolved missing timestamps are excluded.
- Fetch/enrichment failure should not invalidate an otherwise good structured-feed item.

## Game-story threads

Game coverage has its own lane because the product goal is to show several useful writers without scattering repetitive game stories across the main feed.

Core game writers:

- Andrew Baggarly
- Alex Pavlovic
- Shayna Rubin
- Susan Slusser
- Justice delos Santos
- John Shea
- Maria Guardado

Rules:

1. Group game stories by Pacific "baseball day" plus opponent.
2. If any core writer has an eligible story when a new thread is first created, the **earliest publication timestamp among core writers** gets root/top billing. Author tier does not override publication time for this root choice.
3. Other eligible game stories follow as chronological replies.
4. If no core writer is available at creation time, use the existing quality fallback.
5. Once a Bluesky root is live, never replace it later; an earlier-discovered/stronger story found later simply appends.
6. Persist root and latest-parent refs in `state.json` so later scheduled runs can append to the same thread.
7. Do not impose the standalone one-source cap inside a game thread.
8. Do not cross-publisher story-dedupe game-thread articles; multiple perspectives are the point. Exact URL dedupe still applies.

Known edge case: grouping is a Pacific-day + opponent heuristic, not an MLB game-ID system, so doubleheaders are the main structural edge case. Using MLB schedule game IDs would be a future robustness improvement if needed.

## Bluesky presentation

All posted articles are **headline-first**. The reader should see the story before publication/byline metadata. A normal standalone post is:

```text
Example Giants headline
NBC Sports Bay Area · Alex Pavlovic
Read at www.nbcsportsbayarea.com →
```

The Athletic is labeled in the metadata line but links to the actual destination hostname:

```text
Example headline
The Athletic ($) · Andrew Baggarly
Read at www.nytimes.com →
```

Game story text keeps `Game recap ·` on the metadata line:

```text
Giants’ Turner Hill delivers go-ahead RBI in major-league debut
Game recap · SF Chronicle · Shayna Rubin
Read at www.sfchronicle.com →
```

The final line deliberately separates presentation from the rich-text link facet. `Read at ` and ` →` are plain text; **only the exact article destination hostname** is linked to the direct publisher article. Bluesky's client warns when visible linked text does not represent the destination host, which caused the earlier `Read at SF Chronicle →` / `Read at NBC Sports Bay Area →` labels to open a "Leaving Bluesky" confirmation dialog. Matching the linked text to the exact hostname avoids that mismatch warning while preserving direct publisher URLs.

If a usable article image is available, it is uploaded as a **native Bluesky image embed**, including its aspect ratio. If the image fetch/upload fails, the post remains text + the clickable hostname. Image failures are presentation failures, not selection failures.

**Do not use Bluesky external link cards for article presentation.** Earlier iterations produced duplicated headlines, raw URL fallbacks when the card title was blank, and redundant publisher footer bars when the title was replaced by the outlet name. Native image + direct hostname link is the current chosen presentation.

Use the display name **SF Chronicle**, not the full `San Francisco Chronicle`, in the source/author line. The article headline belongs first in post text for both standalone and game stories.

Last-mile image/metadata fetches happen after selection. They are enhancements and should remain non-blocking.

## State

`state.json` is production data, not a fixture to casually reset. Its important current concepts are:

- `posted_urls` — canonical URL history for exact dedupe
- `posted_stories` — recent story metadata for historical story/event dedupe
- `game_threads` — Bluesky root/latest-parent refs and game identity so later runs can append

A dry run must not mutate this state. Any cleanup of state must preserve posted-history and live thread refs.

## Validation standard

Before merging a behavioral change, use `.github/workflows/v2-structured-probe.yml`. It runs:

- deterministic story/game/radar/runtime unit tests
- structured live-source discovery probe
- contained core-writer radar probe
- realistic selection using a copy of production state
- clean-slate duplicate-choice simulation
- full `DRY_RUN=1` V2 bot against production state and verifies byte-for-byte state is unchanged
- Athletic feed/metadata probes
- source diagnostics

Do **not** manually trigger a live Bluesky production run merely to test code. Maintenance/CI should remain non-posting unless a live post is explicitly requested.

## Change philosophy

- Work in small, verifiable batches rather than large rewrites.
- Reuse mature public feeds/libraries before inventing parsers.
- Prefer deterministic explainable rules over opaque scoring/ML for this scale.
- Keep each publisher adapter small and isolated.
- A broken source should not break other sources.
- A failed page fetch should not block a valid core article.
- Preserve direct publisher URLs; do not post Google News wrapper URLs.
- Keep the account selective; adding a source is not automatically better if it adds commodity content.
- Protect production state and avoid live-post side effects during development.

## Files a new session should inspect first

1. `README.md`
2. this file
3. `docs/ARCHITECTURE.md`
4. `docs/OPERATIONS.md`
5. `v2_bot.py`
6. `v2_authors.py`
7. `v2_story.py`
8. `v2_game_threads.py`
9. `v2_selector.py`
10. `.github/workflows/giants-news-bot.yml`

If those agree, they are the current source of truth. Old chat history should not override current code.
