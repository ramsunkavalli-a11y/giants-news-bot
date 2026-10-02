# SF Giants News Bot

Automated, curated San Francisco Giants news feed for Bluesky. The bot discovers stories from a small set of structured publisher sources, filters for Giants relevance and editorial value, deduplicates overlapping coverage, groups game stories into threads, and posts direct publisher links.

**Production is V2.** The entrypoint is `v2_bot.py`. Older crawler/scoring code has been retired and is not part of the current system.

## Production at a glance

- **Entrypoint:** `python v2_bot.py`
- **Workflow:** `.github/workflows/giants-news-bot.yml`
- **Persistent state:** `state.json` (posted history, game-thread refs, and a bounded run-health heartbeat)
- **Baggarly checks:** official author RSS every 15 minutes; Giants stories bypass the routine daily budget (see [fast posting](docs/FAST_POSTING.md))
- **Season mode:** production defaults to `offseason`; repository variable `SEASON_MODE=inseason` restores game-season cadence and limits
- **Standalone cap:** offseason 2/run and 6 routine stories/Pacific day; confirmed moves/injury news bypass the daily routine cap; inseason 3/run
- **Standalone freshness:** 72 hours
- **Game-story freshness:** 30 hours
- **Dry run:** never mutates production state or posts to Bluesky
- **Offseason weekday cadence, Pacific:** 8:30 AM / 2:30 PM / 7:30 PM
- **Offseason weekend cadence, Pacific:** 8:30 AM / 1:30 PM / 5:30 PM
- **Inseason late checks:** additionally 11:30 PM weekdays / 10:30 PM weekends

GitHub cron is UTC, so the production workflow schedules both PDT and PST equivalents and uses an `America/Los_Angeles` gate to keep those local times stable through daylight-saving changes.

## Current source strategy

The design principle is **structured discovery, own the last mile**. Prefer a publisher's official RSS/feed or clean team/category page rather than maintaining a universal crawler.

| Publication | Production discovery |
| --- | --- |
| SF Standard | Dedicated San Francisco Giants tag RSS |
| The Athletic | Giants RSS + official Andrew Baggarly author RSS |
| MLB.com | Giants RSS; Maria Guardado bylines only |
| SFGATE | Giants RSS |
| FanGraphs | Giants category RSS |
| NBC Sports Bay Area | Giants news + analysis pages |
| KNBR | Giants-only Executive Show playlist/RSS via Omny |
| SF Chronicle | Targeted core-writer radar |
| Mercury News | Targeted core-writer radar |

The SF Standard's broad Sports RSS is intentionally not used because it mixes Giants coverage with Warriors, Valkyries, 49ers, and other local sports. The dedicated Giants tag feed establishes team relevance at the source boundary and avoids relying on generic `San Francisco` text matching.

MLB.com's Giants feed mixes team beat reporting with staff packages, national prospect content, promotional streams, and unsigned commodity pages. Production therefore treats **Maria Guardado as the only eligible MLB.com byline**; other MLB.com authors and unsigned feed items are low-value at the source-classification boundary.

Chronicle and Mercury pages are unreliable from GitHub runners, so the bot uses tightly scoped Google News RSS queries only for named writers at those publishers, decodes the wrapper URL, verifies the publisher domain, and then sends the result through the same V2 filters as every other candidate. Mercury coverage includes Justice delos Santos, Evan Webeck, and Dieter Kurtenbach; each Mercury result must also show “Giants” in the article headline or URL path, and explicit coverage of another local sport is rejected. Co-bylines are allowed when the targeted writer is explicitly one of the visible authors. **Broad Google News search is diagnostic only and is not a production source.**

The KNBR integration is deliberately narrow: only the Giants playlist of **The Executive Show** is active. It is not a general KNBR scraper and should not surface 49ers Executive Show episodes.

Baseball America and Associated Press remain trusted/interesting publishers but are not currently active production discoverers.

## Pipeline

```text
structured discovery
        ↓
Giants relevance + editorial-value classification
        ↓
exact URL + story/event dedupe
        ↓
┌──────────────────────────┬────────────────────────────┐
│ standalone news          │ game-story lane            │
│ event/reporting +        │ MLB schedule/gamePk match  │
│ differentiated analysis  │ root + threaded replies    │
│ one source/run           │ separate from cap          │
│ max 3/run                │ heuristic fallback         │
└──────────────────────────┴────────────────────────────┘
        ↓
optional image/metadata enrichment
        ↓
Bluesky post: headline + source/author + publisher-domain link
        ↓
native image when available
        ↓
state + diagnostics
```

A failed article-page fetch must not turn an otherwise valid structured-feed item into a rejection. Page fetches are primarily last-mile enrichment and targeted metadata work, not the foundation of discovery.

## Editorial behavior

The bot is intended to surface original reporting, breaking news, transactions, injuries, prospect coverage, meaningful analysis/features, and trusted beat reporting. It downweights or rejects commodity score recaps, broad multi-team rankings/listicles, recurring chats, mailbag submission prompts, weekly cartoons, promo/stream/highlight pages, video-only content, and derivative articles that mainly summarize another outlet.

Cross-publisher duplicates are clustered deterministically. For event-driven news such as a call-up, the selector can keep **one event/reporting representative plus one genuinely differentiated analysis representative**. For comparable routine event reporting, recent publication representation over a 14-day window is used as a tie-breaker so the same outlet does not automatically win every transaction. Meaningful quality gaps and a substantial early-reporting lead still override that rotation.

The best representative(s) of an event are chosen **before** the one-source-per-run diversity rule is applied; the selector does not fall through to a weaker duplicate merely to fill a slot.

## Offseason approach

In `offseason` mode, confirmed roster moves, contracts, coaching changes, and injury news get first consideration, followed by Giants prospects/scouting, Fall League and winter-ball development, arbitration, Rule 5/40-man decisions, pitching/payroll plans, and season reviews. Other useful features remain eligible. Hypothetical trade proposals and mock trades are excluded; attributed market reporting keeps its original headline qualifiers. There is no minimum posting quota.

Keep the existing 72-hour freshness window and source/role diversity. Season reviews stay in the standalone lane, even when a feed calls them “what we learned.” Recent game coverage still follows the existing schedule-grounded routing. FanGraphs items need explicit Giants/San Francisco evidence in the headline, slug, or first 600 characters of the structured summary; a team tag, author, or search query is insufficient. Mixed-team scouting with explicit Giants evidence remains eligible.

Mode is explicit because a generic MLB calendar cannot determine whether the Giants are still playing in October. Set the GitHub repository variable `SEASON_MODE` to `inseason` when Giants game coverage resumes; use `offseason` for this period. Local runs default to `inseason` for compatibility; pass `SEASON_MODE=offseason` to test current production policy. Manual trusted stories retain priority and may bypass the routine daily cap, while URL/event history remains enforced.

## Bluesky presentation

Stories are **headline-first** so the reader sees the news before the metadata. Standalone example:

```text
Example Giants headline
MLB.com · Maria Guardado
Read at www.mlb.com →
```

Game coverage keeps the game label on the metadata line:

```text
Giants’ Turner Hill delivers go-ahead RBI in major-league debut
Game recap · SF Chronicle · Shayna Rubin
Read at www.sfchronicle.com →
```

Executive Show audio uses the same hierarchy but an audio verb:

```text
Buster Posey discusses the Giants' young pitching
KNBR · The Executive Show
Listen at omny.fm →
```

On the final line, only the exact destination hostname is a Bluesky rich-text link to the direct publisher/audio URL. Matching the visible linked hostname to the destination avoids Bluesky's external-link mismatch warning. If a usable image is available, it is uploaded as a **native Bluesky image** beneath the text. The bot deliberately does **not** use external link cards: they caused duplicate headlines, raw-URL fallbacks, and redundant publisher footer boxes. If an image is missing, invalid, too large, or its upload fails, the post remains text + clickable publisher-domain link only.

Use the display name **SF Chronicle** in the metadata line. The Athletic is displayed as **The Athletic ($)**.

## Game coverage

Game stories use a separate lane so readers can get several useful perspectives without filling the main feed with disconnected recap posts.

- The bot queries the free MLB StatsAPI schedule for Giants games and, when the opponent can be identified, assigns coverage to the closest actual previously started game and its `gamePk`.
- That prevents a delayed next-day story from inventing a new game thread and gives doubleheaders distinct identities.
- When a headline explicitly says Game 1/Game 2, opener, or nightcap, that cue overrides arrival time so postgame stories published after a doubleheader do not collapse into the second game.
- If schedule matching is unavailable, the older Pacific baseball-day + opponent heuristic remains a nonblocking fallback.
- Existing legacy date/opponent thread keys are reused when a new `gamePk` match points to the same game, so live Bluesky thread roots are preserved during migration.
- Core game writers are Andrew Baggarly, Alex Pavlovic, Shayna Rubin, Susan Slusser, Justice delos Santos, John Shea, and Maria Guardado.
- If at least one core writer is available when a new game thread is created, the **earliest-published core-writer story** gets the root post.
- Other eligible game stories become chronological replies.
- Once a Bluesky thread root exists, it is never replaced; later discoveries append to that thread.
- Root/parent Bluesky refs are persisted in `state.json` so later runs can continue the same thread.
- `run_history` in `state.json` keeps the latest 100 production heartbeats, including source failures, selected counts, schedule-grounded counts, and failed-post errors.
- Game stories do not count against the 3-story standalone cap.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
DRY_RUN=1 DIAGNOSTICS_ENABLED=1 python v2_bot.py
```

For a production Bluesky post, `BSKY_IDENTIFIER` and `BSKY_APP_PASSWORD` must be present. Do not use a live run as a routine test; the V2 CI workflow exercises live source discovery and a full production-state dry run without posting.

Useful environment variables:

| Variable | Production/default purpose |
| --- | --- |
| `HOURS_BACK` | 72-hour standalone discovery window |
| `GAME_HOURS_BACK` | 30-hour game-story window in production |
| `MAX_POSTS_PER_RUN` | 3 inseason; offseason clamps to at most 2 |
| `SEASON_MODE` | `inseason` or `offseason`; production default is offseason |
| `OFFSEASON_DAILY_LIMIT` | 6 routine standalone stories per Pacific calendar day |
| `FAST_LANE` | `1` scans only Baggarly’s author RSS with six-hour freshness |
| `DRY_RUN` | `1` prints actions without posting or mutating state |
| `STATE_FILE` | Alternate state path for tests/replays |
| `DIAGNOSTICS_ENABLED` | Write selection/discovery diagnostics |
| `DIAGNOSTICS_FILE` | Diagnostics output path |
| `REQUEST_TIMEOUT` | HTTP timeout, default 15 seconds |
| `KEEP_POSTED_DAYS` | State retention window, default 21 days |
| `BSKY_PDS` | Bluesky PDS, default `https://bsky.social` |

### Manual trusted story

The production workflow's **Run workflow** form accepts an optional article URL, exact headline, and byline. URL and headline must be supplied together. The URL must be HTTPS and belong to an approved production publisher; the bot infers the source from the hostname. The story receives first consideration in that run, but exact-URL and same-event history still prevent a duplicate post.

## Repository map

- `v2_bot.py` — production orchestration
- `v2_probe.py` — **production structured article adapters** plus structured-discovery diagnostic entrypoint
- `v2_knbr.py` — dedicated KNBR Executive Show/Omny discovery
- `v2_radar.py` — tightly scoped Chronicle/Mercury core-writer radar
- `v2_editorial.py` — shared season-review, Giants-evidence, and offseason priority rules
- `v2_selector.py` — freshness, historical dedupe, role-aware event selection, comparable-source rotation
- `v2_story.py` — story/event clustering, event-family normalization, news-vs-analysis role logic
- `v2_game_threads.py` — game detection, schedule-aware grouping, root/reply ordering
- `v2_mlb_schedule.py` — MLB StatsAPI schedule adapter
- `v2_authors.py` — author registry and editorial priors
- `bsky_client.py` — Bluesky headline/metadata/link formatting, native image upload/embed, and reply creation
- `models.py` — shared candidate model
- `config.py` — small V2 runtime settings object
- `v2_*_test.py` — deterministic regression tests
- `v2_*_probe.py`, `v2_source_diag.py`, `v2_select_probe.py` — CI diagnostics/simulations; these are intentional, not retired prototypes
- `.github/workflows/giants-news-bot.yml` — production scheduler
- `.github/workflows/v2-structured-probe.yml` — validation workflow on V2 branches and PRs to `main`

## Documentation

A new maintainer or new ChatGPT session should read these in order:

1. [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) — current decisions, priorities, sources, authors, and known caveats
2. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — pipeline and module-level design
3. [`docs/OPERATIONS.md`](docs/OPERATIONS.md) — scheduling, testing, deployment, state, diagnostics, and troubleshooting

The project intentionally favors small, deterministic rules and mature public feeds/parsers over bespoke crawling or paid infrastructure.
