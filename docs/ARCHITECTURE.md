# V2 architecture

## Design goal

Build a reliable SF Giants news/audio → Bluesky pipeline without maintaining a general-purpose web crawler.

The system is split into a **structured discovery layer** and a **small deterministic editorial layer**. Sources can fail independently. Expensive or fragile article-page fetching is reserved for metadata enrichment and narrow fallback cases.

## End-to-end flow

`v2_bot.py` is the production coordinator.

```text
Article adapters ─────────┐
KNBR Executive Show ──────┼─> candidates
Targeted author radar ────┘
                              │
                              ▼
                  source-aware classification
                  + Giants relevance checks
                              │
                              ▼
                    normalize/dedupe URLs
                              │
                    ┌─────────┴────────────┐
                    ▼                      ▼
             standalone lane          game-story lane
                    │                      │
             event clustering          MLB schedule/gamePk
             news vs analysis          match when possible
             comparable rotation       heuristic fallback
             historical dedupe         root + replies
             one source/run                 │
             max N/run                      │
                    └─────────┬────────────┘
                              ▼
                  selected candidates only
                              │
                              ▼
                 optional image enrichment
                              │
                              ▼
              headline + source/author + domain link
              + native Bluesky image if available
                              │
                              ▼
                         Bluesky API
                              │
                              ▼
                   state + diagnostics
```

## 1. Structured discovery

### `v2_probe.py`

Contains the six direct article adapters:

- SF Standard dedicated San Francisco Giants tag RSS
- The Athletic Giants RSS
- MLB.com Giants RSS
- SFGATE Giants RSS
- FanGraphs Giants category RSS
- NBC Sports Bay Area Giants news/analysis listing pages

The SF Standard adapter deliberately uses the Giants tag feed rather than the broad Sports RSS, which mixes Warriors, Valkyries, 49ers, Giants and other local sports. Team relevance is therefore established by the publisher's structured tag boundary instead of generic `San Francisco` text matching.

Each adapter returns `Article` objects with normalized source, title, direct URL, publication time, author when available, summary, access and classification metadata.

Adapters remain publisher-specific. Repair a broken adapter rather than building a global crawler abstraction.

### `v2_knbr.py`

A separate audio adapter follows the Giants-only Omny playlist for **KNBR The Executive Show**. The code discovers the RSS endpoint from the playlist page rather than hard-coding a private-looking Omny GUID URL.

The adapter:

- reads the Giants Executive Show playlist/RSS;
- filters for Giants/front-office guests;
- rejects 49ers-only episodes;
- emits `Article`-compatible records with source `KNBR` and author `The Executive Show`.

This is intentionally not broad KNBR discovery.

### `v2_radar.py`

Chronicle/Mercury discovery is isolated because direct access from GitHub runners is unreliable.

Production radar targets named writer/publisher pairs. Google News RSS is a discovery transport; `googlenewsdecoder` resolves wrappers to direct publisher URLs.

Safety constraints:

- exact target domain after decoding;
- exact author query;
- small result cap per author;
- URLs matching multiple target-author queries are rejected as ambiguous;
- contradictory visible author metadata rejects;
- co-bylines are accepted when the targeted core writer is explicitly included;
- challenge/blank metadata does not independently veto a tightly attributed result;
- broad Google News never enters production selection.

## 2. Candidate/article models

`v2_probe.Article` is the discovery representation. `models.Candidate` is the posting/runtime representation.

Runtime-relevant fields include source, direct URL, title, timestamp, author, summary, access status, optional image URL and discovery path. Do not restore fields solely because the retired crawler once had them.

## 3. Editorial classification

Desired high-value material includes original reporting, breaking news, transactions, injuries, prospect coverage, substantive analysis/features, trusted beat work and direct Giants executive interviews.

Medium/game-specific material includes genuinely authored postgame analysis and result-heavy gamers that belong in the game lane.

Low-value material includes commodity recaps, generic multi-team pieces/rankings, recurring evergreen pages, promo/stream/highlight pages, video-only pages, press-conference clips without reporting, and derivative pieces that mainly summarize another outlet.

The selector also maintains a last-mile safety filter for known broad/highlight title patterns so an adapter classification miss does not automatically become a post.

## 4. Story/event clustering — `v2_story.py`

The system recognizes the same news event without treating every article about the same player as identical.

Normalization includes event families. In particular, `promoted`, `called up`, `call-up`, `gets the call`, and similar forms can resolve to the same call-up event when an identifying subject overlaps.

### Story roles

Same event does not necessarily mean same reader value. `story_role(article)` currently distinguishes:

- `news` — event/reporting/roster/organizational coverage;
- `analysis` — deeper interpretation/scouting/style/repertoire work.

Grant Brisbee and FanGraphs are simple analysis priors, with additional title cues. This is intentionally lightweight rather than an ML classifier.

### Preference tuple

Within a role, the basic editorial preference tuple remains:

1. author preference;
2. light source preference;
3. named byline;
4. recency.

## 5. Standalone selector — `v2_selector.py`

The selector handles canonical URL dedupe, recent-state history, freshness, missing-time enrichment, high-quality gating, role-aware event selection, publication diversity and the run cap.

### One news + one analysis representative

For one event cluster, the selector may keep:

- at most one news/reporting representative;
- at most one differentiated analysis representative.

Other same-role versions are duplicate-suppressed. This means three routine versions of a call-up do not all post, while a materially deeper analysis triggered by that call-up can survive alongside the chosen news version.

Historical event dedupe is role-aware. A previously posted news version does not automatically suppress a later analysis version; a same-event analysis can suppress another analysis version.

### Comparable-story rotation

For event-driven news candidates that are close enough in author preference, a deterministic source-rotation tie-breaker uses recent standalone publication counts over 14 days. Lower-represented publications are preferred among comparable candidates.

Rotation is not random and does not trump obvious reporting advantages:

- candidates more than one author-prior tier below the top are not considered comparable;
- an early-reporting lead of at least 90 minutes wins before rotation;
- normal editorial preference breaks remaining ties.

After event representatives are chosen, the existing one-publication-per-standalone-run rule is applied. Production `MAX_POSTS_PER_RUN` remains 3.

## 6. Game-story lane — `v2_game_threads.py`

### Detection

Title/summary patterns identify genuine game recaps and postgame analysis. Low-value highlights/promos remain rejected.

### Schedule grounding — `v2_mlb_schedule.py`

The schedule adapter uses the free MLB StatsAPI schedule endpoint for Giants team ID **137** and returns game metadata including `gamePk`, official date, start time and opponent.

For eligible game coverage, `v2_game_threads.py` attempts to:

1. extract the opponent from title/summary;
2. find the closest matching actual Giants game that had already started;
3. require it to fall within a conservative 48-hour window;
4. use `game:{gamePk}` as the new stable thread key.

This prevents delayed follow-up articles from inventing phantom next-day games and separates doubleheaders.

Schedule access is nonblocking. If the API cannot be reached or the opponent cannot be identified, grouping falls back to the Pacific baseball-day + opponent heuristic.

### Legacy-thread migration

Production already contains date/opponent thread keys. Before creating a new gamePk-rooted thread, `v2_bot._existing_thread_key` checks for an existing legacy `game:YYYY-MM-DD:opponent` thread for the same game. That preserves live root/parent refs rather than splitting an existing thread during migration.

### Root ordering

`v2_authors.CORE_GAME_WRITERS` defines preferred root writers. When creating a new thread:

- if core writers are present, earliest-published core writer gets the root;
- remaining stories are chronological replies;
- if no core writer is present, normal quality fallback is used.

A live root is immutable. Game posts are outside standalone source diversity and the standalone cap.

## 7. Optional enrichment

Selected candidates can be enriched with author metadata, missing publication time and `og:image`/`twitter:image`. This is last-mile work. Failure degrades presentation rather than suppressing a valid structured item.

Chronicle/Mercury challenge pages make image absence expected. KNBR may provide Omny artwork; it is treated as optional like any other image.

## 8. Bluesky — `bsky_client.py`

Responsibilities:

- login/session;
- headline-first formatting;
- source/author metadata, including `Game recap ·` for game stories;
- exact-hostname rich-text link facets;
- `Read at` for article sources;
- **`Listen at` for KNBR audio**;
- native image upload/embed with aspect ratio;
- reply root/parent payloads.

Article example:

```text
Example Giants headline
Mercury News · Justice delos Santos
Read at www.mercurynews.com →
```

Executive Show example:

```text
Buster Posey discusses the Giants' young pitching
KNBR · The Executive Show
Listen at omny.fm →
```

Only the exact hostname is linked. External link cards are intentionally not used.

## 9. State

Important persistent concepts:

```json
{
  "posted_urls": {},
  "posted_stories": [],
  "game_threads": {}
}
```

`posted_urls` prevents exact reposts. `posted_stories` supports role-aware historical event dedupe and recent source-count rotation. `game_threads` stores Bluesky refs and game identity; new entries can include `game_pk`.

State changes only after live posting. Dry-run validation uses a copy and verifies byte-for-byte immutability.

## 10. Failure isolation

- each discoverer runs independently;
- a KNBR failure cannot disable article sources;
- radar failure cannot disable direct sources;
- StatsAPI failure falls back to heuristic game grouping;
- image enrichment failure cannot reject selection;
- bad visible metadata is ignored/rejected as appropriate;
- URL/story history prevents accidental reposts.

## 11. Module map

| File | Role |
| --- | --- |
| `v2_bot.py` | production orchestration, conversion, enrichment, posting, state |
| `v2_probe.py` | six direct article adapters + structured probe CLI |
| `v2_knbr.py` | Executive Show/Omny audio adapter |
| `v2_radar.py` | blocked core-writer radar |
| `v2_selector.py` | standalone filtering, role-aware dedupe, comparable-source rotation |
| `v2_story.py` | event families, story roles, clustering/preferences |
| `v2_game_threads.py` | game detection, schedule-aware grouping, root ordering |
| `v2_mlb_schedule.py` | MLB StatsAPI schedule adapter |
| `v2_authors.py` | author/source priors and core game writer registry |
| `bsky_client.py` | Bluesky text/link/image/reply posting |
| `models.py` | shared runtime candidate model |
| `config.py` | runtime environment settings |
| `v2_*_test.py` | deterministic regression tests |
| `v2_*_probe.py`, `v2_source_diag.py` | live-source diagnostics used in CI |
| `v2_select_probe.py` | selection simulations against state |

## 12. Architectural guardrails

1. Do not reintroduce a general crawler merely because a publisher is awkward.
2. Do not use broad Google News as production discovery.
3. Do not let page-fetch failure block a valid structured article.
4. Do not replace deterministic event dedupe with embeddings/paid APIs without demonstrated need.
5. Preserve differentiated analysis when it genuinely adds a second reader value to the same event.
6. Use publication rotation only among comparable event-news candidates; it is not a quality override.
7. Do not change a live game-thread root after posting.
8. Keep StatsAPI schedule access nonblocking with a deterministic fallback.
9. Do not make CI/maintenance tests create live Bluesky posts.
10. Do not reset production state as part of code cleanup.
11. Prefer one small adapter per publication/source.
12. Keep product volume selective.
13. Keep direct destination URLs as rich-text links; do not revert to external cards without a specific UI reason.
