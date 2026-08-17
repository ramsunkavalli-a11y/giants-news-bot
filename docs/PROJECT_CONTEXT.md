# Project context / new-session handoff

This is the quickest way for a new maintainer or a new ChatGPT conversation to understand the SF Giants News Bot without reconstructing prior discussions.

## What the product is

A Bluesky bot that posts a curated stream of San Francisco Giants journalism and selected original audio from multiple publications. The goal is **useful coverage, not maximum volume**. It should feel closer to an automatically maintained Giants news desk than an indiscriminate RSS firehose.

The account favors original reporting, breaking news, transactions, injuries, prospect work, meaningful analysis/features, trusted beat reporting, and direct access to Giants decision-makers. It should avoid generic summaries, commodity recaps, broad multi-team rankings/listicles, promo pages, highlights/video-only pages, recurring evergreen content, and derivative articles whose main value is repeating another outlet.

Cost should remain essentially zero while the account is small. Prefer mature public feeds/parsers and deterministic logic over paid APIs, embeddings, or custom crawling infrastructure.

## Current production behavior

Production runs `v2_bot.py` from `.github/workflows/giants-news-bot.yml`.

### Cadence

Pacific local time:

- **Monday–Friday:** 8:30 AM, 2:30 PM, 7:30 PM, 11:30 PM
- **Saturday–Sunday:** 8:30 AM, 1:30 PM, 5:30 PM, 10:30 PM

The weekend shifts earlier because Giants weekends contain more day games and the first postgame stories often arrive in the afternoon. GitHub schedules in UTC; the workflow includes PDT and PST equivalents and an `America/Los_Angeles` gate so local times do not drift across daylight saving.

### Volume/freshness

- Standalone stories: maximum **3 per run**.
- Standalone discovery window: **72 hours**.
- Game-story window: **30 hours**.
- Game-thread replies are separate from the standalone cap.
- A run can legitimately post nothing.
- GitHub Actions scheduled jobs may start several minutes late.

## Source philosophy

**Outsource/use structured discovery; own the last mile.** Do not build another universal crawler unless a compelling source leaves no cleaner option.

### Active direct sources

1. **SF Standard** — official Sports RSS
2. **The Athletic** — Giants RSS
3. **MLB.com** — Giants RSS
4. **SFGATE** — Giants RSS
5. **FanGraphs** — Giants category RSS
6. **NBC Sports Bay Area** — dedicated Giants news and analysis pages
7. **KNBR The Executive Show** — Giants-only Omny playlist/RSS in `v2_knbr.py`

The first six article adapters are implemented in `v2_probe.py`. Despite the filename, those discoverers are production code. KNBR is intentionally separate because it is audio and has different presentation/filtering semantics.

### KNBR Executive Show

The bot does **not** broadly scrape KNBR. It follows the Giants-only playlist for The Executive Show, a recurring baseball-season front-office/manager interview series, and filters away 49ers-only episodes. Typical guests include Buster Posey, Zack Minasian, Tony Vitello, Larry Baer, and other Giants leadership.

Executive Show posts are headline-first like articles, but use:

```text
Episode title
KNBR · The Executive Show
Listen at omny.fm →
```

The existing Thursday 8:30 AM Pacific production poll is expected to catch most Thursday-morning episodes. Do not add a special Thursday poll unless real publication timing shows that this regularly misses them.

### Blocked-source radar

SF Chronicle and Mercury News are important, but their pages/feeds are unreliable from GitHub runners. Production therefore has a deliberately narrow radar in `v2_radar.py` for:

- Susan Slusser — SF Chronicle
- Shayna Rubin — SF Chronicle
- Justice delos Santos — Mercury News

For each writer the bot queries Google News RSS with the exact author + exact publisher domain + Giants, decodes the Google wrapper URL, verifies the resulting publisher domain, limits the results, rejects ambiguous multi-target attribution, and uses visible page metadata as an additional veto when available. A co-byline is valid when the targeted core writer is explicitly one of the visible authors; a contradictory byline still rejects the result.

This is intentionally **not** broad Google News discovery. Broad Google results are diagnostic only.

### Trusted but inactive names

Baseball America and Associated Press remain worth considering for future targeted discovery, but they are not currently active production discoverers.

## Editorial priors

Author preference is useful for choosing among overlapping coverage; it is not blanket permission to publish low-value material.

Current local/beat priors in `v2_authors.py`:

- **Elite:** Andrew Baggarly, Alex Pavlovic, Shayna Rubin
- **Very good:** Susan Slusser, Justice delos Santos
- **Good:** John Shea, Maria Guardado, Alex Simon; Kerry Crowley and Tim Kawakami when Giants-specific
- **Fine:** Grant Brisbee, Evan Webeck
- **National / high-value when Giants-specific:** Jeff Passan, Buster Olney, Jon Heyman, Bob Nightengale, Jon Morosi, Robert Murray, Ken Rosenthal, Evan Drellich

SFGATE has a mild `secondary` publication prior. This is a tie-breaker, not hard suppression.

## Story/event dedupe

The system avoids a global numerical relevance score. Selection is mostly deterministic.

`v2_story.py` normalizes titles, event anchors and useful synonyms. Promotion/call-up language such as `promoted`, `called up`, and `gets the call` belongs to the same event family when the identifying subject matches.

A key product rule is now **same event does not always mean one URL**. The selector distinguishes:

- **news/reporting** — the event, roster implications, quotes, organizational context;
- **analysis** — materially deeper interpretation/scouting/style/repertoire work.

For a single event, the selector can retain at most one strong news/reporting representative and one differentiated analysis representative. Grant Brisbee and FanGraphs are simple current analysis priors; title cues can also identify analysis. The purpose is to avoid three interchangeable call-up posts without suppressing a useful deeper piece triggered by the same transaction.

### Comparable-story rotation

Routine event coverage should not always resolve to the same publication merely because one writer has a slightly higher prior. For comparable event-news candidates:

- use recent standalone publication representation over a **14-day** window as a tie-breaker;
- consider candidates within one author-prior tier of the strongest candidate comparable;
- a substantial early-reporting lead (currently **90 minutes**) overrides rotation;
- larger quality differences still use normal editorial preference.

This is deterministic rotation, not random fairness. Publication diversity is secondary to reader value.

The representative(s) of an event are chosen before the one-source-per-run rule. Historical dedupe is role-aware: a previously posted news version does not automatically suppress a later differentiated analysis version, but another same-event analysis version can be suppressed.

## Standalone selection

- High-quality candidates only.
- Maximum 3/run.
- One standalone story per publication per run.
- One event-news representative plus, when justified, one differentiated analysis representative.
- Broad all-MLB ranking/listicle patterns are rejected at the selector safety boundary even if an adapter misclassifies them.
- Missing timestamps are enriched where possible; unresolved missing timestamps are excluded.
- Fetch/enrichment failure should not invalidate an otherwise good structured-feed item.

## Game-story threads

Game coverage has its own lane because the product goal is to show several useful writers without scattering repetitive game stories across the main feed.

### MLB schedule grounding

`v2_mlb_schedule.py` uses the free MLB StatsAPI schedule endpoint for Giants team ID 137. When a game-story opponent can be identified, `v2_game_threads.py` matches the story to the closest actual Giants game that had already started, within a conservative time window, and uses MLB `gamePk` as the stable new thread identity.

This fixes two important cases:

- a Monday article discussing Sunday's loss should append to Sunday's real thread rather than create a phantom Monday game;
- doubleheaders have separate `gamePk` values instead of one date/opponent bucket.

If MLB schedule access or opponent extraction fails, the older Pacific baseball-day + opponent heuristic remains a nonblocking fallback. During migration, a new gamePk-based group will reuse an existing legacy `game:YYYY-MM-DD:opponent` thread when they refer to the same game, preserving live Bluesky root/parent refs.

Core game writers:

- Andrew Baggarly
- Alex Pavlovic
- Shayna Rubin
- Susan Slusser
- Justice delos Santos
- John Shea
- Maria Guardado

Rules after grouping are unchanged: earliest-published available core writer gets a new root; later eligible stories are chronological replies; a live root is immutable; game stories are outside the standalone cap; cross-publisher game perspectives are intentionally retained.

## Bluesky presentation

All posts are headline-first. Standard article:

```text
Example Giants headline
NBC Sports Bay Area · Alex Pavlovic
Read at www.nbcsportsbayarea.com →
```

The Athletic:

```text
Example headline
The Athletic ($) · Andrew Baggarly
Read at www.nytimes.com →
```

Game story:

```text
Example postgame headline
Game recap · SF Chronicle · Shayna Rubin
Read at www.sfchronicle.com →
```

Executive Show audio uses `Listen at`, not `Read at`.

Only the exact destination hostname is the rich-text link facet. This prevents Bluesky's misleading-label/"Leaving Bluesky" confirmation while preserving the direct publisher URL. If a usable image is available, it is uploaded as a native Bluesky image with aspect ratio. Image failure is cosmetic and nonblocking. Do not restore external link cards without a specific product reason.

## State

`state.json` is production data, not a fixture to casually reset. Important concepts:

- `posted_urls` — canonical URL history
- `posted_stories` — recent story metadata used for historical/event dedupe and recent source representation
- `game_threads` — Bluesky root/latest-parent refs plus game identity; newer entries may also include `game_pk`

A dry run must not mutate this state. Any cleanup must preserve posted history and live thread refs.

## Validation standard

`.github/workflows/v2-structured-probe.yml` runs on V2 development branches and pull requests to `main`. It includes:

- deterministic story/selector/game/schedule/KNBR/radar/runtime tests;
- structured live-source discovery;
- a live KNBR Executive Show adapter probe;
- a live MLB StatsAPI schedule probe;
- contained core-writer radar;
- realistic production-state selection;
- clean-slate duplicate-choice simulation;
- full `DRY_RUN=1` V2 bot against a copy of production state, with byte-for-byte immutability check;
- Athletic and source diagnostics.

Do **not** manually trigger a live Bluesky production run merely to test code.

## Change philosophy

- Work in small, verifiable batches rather than large rewrites.
- Reuse mature public feeds/libraries before inventing parsers.
- Prefer deterministic explainable rules over opaque scoring/ML for this scale.
- Keep publisher adapters small and isolated.
- A broken source should not break unrelated sources.
- A failed page fetch should not block a valid core article.
- Preserve direct publisher URLs; do not post Google News wrapper URLs.
- Keep the account selective; adding a source is not automatically better.
- Protect production state and avoid live-post side effects during development.

## Files a new session should inspect first

1. `README.md`
2. this file
3. `docs/ARCHITECTURE.md`
4. `docs/OPERATIONS.md`
5. `v2_bot.py`
6. `v2_selector.py`
7. `v2_story.py`
8. `v2_game_threads.py`
9. `v2_mlb_schedule.py`
10. `v2_knbr.py`
11. `v2_radar.py`
12. `.github/workflows/giants-news-bot.yml`

If those agree, they are the current source of truth. Old chat history should not override current code.
