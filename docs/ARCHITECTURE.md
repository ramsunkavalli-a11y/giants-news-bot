# V2 architecture

## Design goal

Build a reliable SF Giants news → Bluesky pipeline without maintaining a general-purpose web crawler.

The system is deliberately split into a **structured discovery layer** and a **small deterministic editorial layer**. Publishers can fail independently. Expensive or fragile article-page fetching is reserved for metadata enrichment and narrow fallback cases.

## End-to-end flow

`v2_bot.py` is the production coordinator.

```text
Direct source adapters ─┐
                       ├─> Article candidates
Targeted author radar ─┘
                              │
                              ▼
                  source-specific classification
                  + Giants relevance checks
                              │
                              ▼
                    normalize/dedupe URLs
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             standalone lane       game-story lane
                    │                   │
             story/event cluster    group by game
             choose best version    choose/order root+replies
             historical dedupe      exact-URL dedupe
             one source/run              │
             max N/run                   │
                    └─────────┬─────────┘
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

This module contains the six direct production adapters even though its name predates the final architecture. Keep that in mind before treating it as disposable diagnostics code.

The adapters use the cleanest free structured source available for each publisher:

- SF Standard Sports RSS
- The Athletic Giants RSS
- MLB.com Giants RSS
- SFGATE Giants RSS
- FanGraphs Giants category RSS
- NBC Sports Bay Area Giants news/analysis listing pages

Each adapter returns `Article` objects with normalized fields such as source, title, direct URL, publication time, author when available, summary, and editorial classification metadata.

Adapters should remain small and publisher-specific. If a source format changes, repair that adapter rather than creating a global parsing abstraction that makes unrelated sources more fragile.

### `v2_radar.py`

Chronicle/Mercury discovery is isolated because direct access from GitHub runners is unreliable.

Production radar targets only named writer/publisher pairs. Google News RSS is used as a discovery transport, then `googlenewsdecoder` resolves each wrapper to the direct publisher URL.

Safety constraints:

- exact target domain must match after decoding
- exact author query is used
- only a small number of items per author are inspected
- URLs that appear in multiple author queries are rejected as ambiguous
- contradictory visible author metadata rejects the result
- an anti-bot/challenge page with no useful metadata does not by itself veto an otherwise tightly attributed result
- broad Google News results never enter production selection

This radar is best viewed as a contained adapter for blocked publishers, not a general search engine.

## 2. Candidate/article models

`v2_probe.Article` is the discovery-side representation. `models.Candidate` is the posting/runtime representation used by the selector and Bluesky client.

The runtime needs, at minimum:

- source/publication
- direct article URL and canonical/post URL
- title
- publication timestamp
- author
- summary
- access/paywall status
- image URL when available
- how the item was discovered

Do not add fields solely because an old crawler used them. Add data when a current V2 decision actually consumes it.

## 3. Editorial classification

Source adapters perform source-aware classification rather than relying on one giant global relevance score.

### Desired high-value material

- original reporting
- breaking news
- transactions/roster moves
- injuries and meaningful rehab updates
- prospect/farm reporting
- substantive analysis
- features/interviews with real Giants information
- trusted beat reporting

### Medium/game-specific material

Genuinely authored postgame analysis and result-heavy gamers may be useful in the game lane even when they would be too repetitive as separate standalone feed items.

### Low-value material

Common rejection/downweight examples:

- commodity wire/score recaps
- generic multi-team pieces
- recurring evergreen/team-info pages
- streaming/promo pages
- highlight/video-only pages
- press-conference clips without meaningful article reporting
- podcast snippets presented as articles
- derivative writeups that mainly repeat another outlet's reporting

Author priors do not rescue low-value article types. The content gate comes first.

## 4. Story/event clustering — `v2_story.py`

The system needs to recognize that two different URLs/headlines can describe one news event without accidentally treating every article about one player as the same story.

The clustering logic is deterministic:

- normalize headline text
- remove generic Giants/baseball boilerplate
- recognize event anchors and useful synonyms
- require a meaningful subject/event overlap or sufficiently strong token similarity
- restrict matching to a recent time window

Examples intended to cluster:

- multiple reports of the same Matt Chapman surgery
- multiple reports of one transaction/signing/trade
- the same retirement announcement
- the same All-Star hosting announcement

Examples intended **not** to cluster:

- a Logan Webb game story and a separate Logan Webb development/analysis story
- a player's award story and a later injury story

### Duplicate winner

For standalone duplicates, the preference tuple is intentionally simple:

1. author preference
2. light source preference
3. named byline vs no useful byline
4. publication recency

Author tiers are descriptive editorial priors, not a universal numerical score.

### Diversity rule ordering

Cluster → choose best article → apply source diversity.

Do not choose a lower-quality version of the same event merely because its publication has not yet appeared in the current run.

## 5. Standalone selector — `v2_selector.py`

The standalone selector handles:

- canonical URL dedupe
- recent-state URL history
- recent `posted_stories` title/event history
- freshness
- missing-time enrichment where possible
- high-quality requirement
- event-level winner selection
- one publication per standalone run
- `MAX_POSTS_PER_RUN`

The production cap is 3.

Historical migration logic may inspect recent posted URLs that predate richer story-memory records. That is a compatibility mechanism for existing production state, not a reason to restore the old crawler.

## 6. Game-story lane — `v2_game_threads.py`

A game story is intentionally treated differently from normal news.

### Detection

Title patterns identify genuine game recaps/postgame analysis. Low-value highlights/promos still fail the editorial gate.

### Grouping

Stories are assigned to a Pacific baseball day using an `America/Los_Angeles` time shift and grouped with opponent information where available.

An existing `unknown`-opponent thread can be reused when later data identifies the opponent.

### Root ordering

`v2_authors.CORE_GAME_WRITERS` defines the writers eligible for the preferred root rule.

When creating a new thread:

- if one or more core writers are present, choose the **earliest-published core writer**, regardless of the normal elite/very-good/good author tier
- order remaining eligible stories chronologically
- if no core writer is present, use the normal quality fallback

After a root has been posted, it is immutable. A newly discovered earlier core-writer story becomes a reply.

### Persistence

`state.json.game_threads` stores enough Bluesky refs to send future stories as replies to the correct root and latest parent.

Game-story posting is not constrained by standalone publication diversity or the standalone 3-post cap.

## 7. Optional enrichment

Selected candidates can be enriched with:

- author metadata
- article publication time when missing
- `og:image` / `twitter:image`
- comparable summary metadata when useful diagnostically

This is last-mile work. A failed enrichment request should degrade presentation, not suppress a good structured-feed story.

Chronicle/Mercury are especially likely to return challenge pages, so image/metadata absence is expected and presentation code must fail gracefully.

## 8. Bluesky — `bsky_client.py`

Responsibilities:

- login/session
- headline-first text formatting
- source/author metadata formatting, with `Game recap ·` for game-thread posts
- `Read at <exact hostname> →` final line pointing to the direct article URL
- rich-text facet applied only to the exact hostname so Bluesky does not show a link-label mismatch warning
- optional remote image download + Bluesky blob upload
- native `app.bsky.embed.images` payload when an image is available
- reply root/parent payloads

Standalone example:

```text
Example Giants headline
Mercury News · Justice delos Santos
Read at www.mercurynews.com →
```

Game example:

```text
Example postgame headline
Game recap · SF Chronicle · Shayna Rubin
Read at www.sfchronicle.com →
```

The headline comes first because the article itself is the reader-facing hook; publication and byline are supporting metadata. The final link line uses the article's actual hostname. Only that hostname is linked, while `Read at ` and ` →` remain plain text.

The system **does not use `app.bsky.embed.external` for article presentation**. Earlier external-card iterations had three UI problems: duplicate headline/summary content, a raw article URL when the card title was blank, and a redundant publisher footer when the title was replaced with the source name. Native image + direct hostname link is the chosen presentation.

When no image is available, the same text/link post is created without an image embed. Image failure is therefore cosmetic and non-blocking.

## 9. State

The important persistent state concepts are:

```json
{
  "posted_urls": {},
  "posted_stories": [],
  "game_threads": {}
}
```

`posted_urls` prevents exact reposts, `posted_stories` supports richer historical story dedupe, and `game_threads` allows later scheduled runs to continue an existing Bluesky thread.

State is committed back to the repository after a live scheduled run only when it changes.

Dry-run validation uses a copy of production state and verifies the copy is unchanged after execution.

## 10. Failure isolation

Core robustness principles:

- each discoverer runs independently
- one source throwing an exception should not abort the whole discovery pass
- radar failure should not disable direct sources
- image enrichment failure should not reject selection
- bad metadata should be ignored rather than posted when it is obviously publisher/challenge boilerplate
- exact URL/story history should prevent accidental reposts across runs

## 11. Module map

| File | Role |
| --- | --- |
| `v2_bot.py` | production orchestration, conversion, enrichment, posting, state |
| `v2_probe.py` | direct production source adapters + structured probe CLI |
| `v2_radar.py` | blocked core-writer radar |
| `v2_selector.py` | standalone filtering/dedupe/selection |
| `v2_story.py` | event/story keys and duplicate comparison |
| `v2_game_threads.py` | game detection/grouping/root ordering |
| `v2_authors.py` | author/source priors and core game writer registry |
| `bsky_client.py` | Bluesky headline/metadata/link formatting, native image embed, API posting |
| `models.py` | shared runtime candidate model |
| `config.py` | runtime environment settings |
| `v2_*_test.py` | regression tests |
| `v2_*_probe.py` / `v2_source_diag.py` | live-source diagnostics used in CI |
| `v2_select_probe.py` | selection simulations against state |

## 12. Architectural guardrails

When changing the project, preserve these unless there is a deliberate product decision to replace them:

1. Do not reintroduce a general crawler merely because a publisher is awkward.
2. Do not use broad Google News as production discovery.
3. Do not let page-fetch failure block a valid structured article.
4. Do not replace deterministic event dedupe with embeddings/paid APIs without a demonstrated need.
5. Do not let source diversity pick a weaker duplicate.
6. Do not change a live game-thread root after posting.
7. Do not make CI/maintenance tests create live Bluesky posts.
8. Do not reset production state as part of code cleanup.
9. Prefer one small adapter per publication.
10. Keep product volume intentionally selective.
11. Keep the article URL as a direct publisher rich-text link; do not revert to external cards without a specific UI reason.
