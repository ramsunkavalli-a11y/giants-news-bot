# Operations

## Production workflow

Production is `.github/workflows/giants-news-bot.yml` on `main`.

The workflow:

1. evaluates the Pacific-time cadence gate;
2. checks out `main`;
3. installs Python 3.11 dependencies;
4. runs `python v2_bot.py`;
5. uploads `diagnostics.json`;
6. commits `state.json` back to `main` only if it changed.

Required GitHub Actions secrets:

- `BSKY_IDENTIFIER`
- `BSKY_APP_PASSWORD`

The default Bluesky PDS is `https://bsky.social` and can be overridden with `BSKY_PDS`.

## Schedule

All intended times are **America/Los_Angeles** local time.

### Monday–Friday

- 8:30 AM
- 2:30 PM
- 7:30 PM
- 11:30 PM

### Saturday–Sunday

- 8:30 AM
- 1:30 PM
- 5:30 PM
- 10:30 PM

The earlier weekend windows better match day-game reporting. GitHub Actions cron is UTC, so production registers PDT and PST equivalents and checks the current Pacific offset. The inactive duplicate exits before checkout/install.

Scheduled jobs can begin several or even tens of minutes late; scheduler delay is not automatically a bot failure.

The existing Thursday 8:30 AM poll is the expected collection point for most KNBR Executive Show episodes. Do not add a Thursday-specific production poll unless observed Omny publication timing demonstrates a recurring miss.

## Production environment

```text
HOURS_BACK=72
GAME_HOURS_BACK=30
MAX_POSTS_PER_RUN=3
DIAGNOSTICS_ENABLED=1
DIAGNOSTICS_FILE=diagnostics.json
```

Other environment-backed settings are in `config.py`.

## Safe local/test execution

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
DRY_RUN=1 DIAGNOSTICS_ENABLED=1 HOURS_BACK=72 GAME_HOURS_BACK=30 MAX_POSTS_PER_RUN=3 python v2_bot.py
```

Use an alternate state file for state experiments:

```bash
STATE_FILE=/tmp/test-state.json DRY_RUN=1 python v2_bot.py
```

A dry run must never post to Bluesky or mutate state.

## Validation workflow

`.github/workflows/v2-structured-probe.yml` runs on V2/maintenance branch pushes **and pull requests to `main`**. The pull-request trigger is the inspectable pre-merge gate; behavioral PRs should not merge while it is failing.

The suite includes:

1. deterministic story clustering tests;
2. role-aware selector/rotation tests;
3. game-thread and doubleheader tests;
4. MLB schedule parsing/migration tests;
5. KNBR Executive Show/filter/link-format tests;
6. radar/co-byline tests;
7. Bluesky runtime/image/link/state tests;
8. live structured article discovery;
9. live KNBR Executive Show/Omny probe;
10. live MLB StatsAPI schedule probe;
11. contained Chronicle/Mercury radar probe;
12. copy of current production `state.json`;
13. realistic selection simulation against production state;
14. clean-slate duplicate-choice simulation;
15. full V2 dry run against copied production state;
16. byte-for-byte verification that dry-run state is unchanged;
17. Athletic RSS/metadata probes and source diagnostics;
18. artifact upload of all probe outputs.

## Deployment

There is no separate server. Merging to `main` changes what the next scheduled GitHub Actions run executes.

Normal process:

1. branch from current `main`;
2. make a coherent change;
3. run/inspect the exact-head V2 validation workflow;
4. review live adapter and production-state diagnostics;
5. confirm dry-run state is unchanged;
6. merge;
7. let the next scheduled production run exercise the code naturally.

Do **not** manually dispatch production just to prove a merge worked. A manual production dispatch can create live Bluesky posts.

## State management

`state.json` is production data. Important fields:

- `posted_urls` — exact/canonical URL history;
- `posted_stories` — recent story metadata used for role-aware event dedupe and recent publication representation;
- `game_threads` — live Bluesky root/latest-parent refs and game identity; newer entries can include MLB `game_pk`.

Never casually reset state, delete posted history to make tests pass, or remove live game-thread refs. Schema migrations must preserve dedupe history and Bluesky refs and should be validated on a copy first.

### gamePk migration

Newly schedule-grounded game groups use `game:{gamePk}`. Existing production threads may still use `game:YYYY-MM-DD:opponent`. Runtime lookup intentionally reuses a legacy thread when its date/opponent matches the newly grounded game, so no state rewrite is required just to migrate identifiers.

## Diagnostics

Production uploads `diagnostics.json` for every real scan that passes the cadence gate. V2 CI uploads a broader `v2-structured-probe` artifact containing source, KNBR, schedule, radar, selection, runtime, Athletic and source-diagnostic outputs.

Absence of a post can mean:

- workflow has not run yet;
- cadence gate skipped an inactive UTC twin;
- no new candidate was discovered;
- candidate was stale or already posted;
- same event/role was already posted;
- candidate lost same-role event dedupe;
- a different role from the same event was retained instead;
- source diversity or run cap suppressed it;
- candidate is low-value or too broad;
- candidate belongs in a game thread.

## Troubleshooting: no post appeared

1. Confirm the scheduled workflow actually ran.
2. Confirm `run_bot=true` at the Pacific cadence gate.
3. Inspect the first exception if `Run V2 bot` failed; one source error normally should not abort discovery.
4. Inspect diagnostics for discovered counts, rejection reasons, selected standalone items and game-thread actions.
5. Do not loosen filters merely because a run was quiet.

## Troubleshooting: source disappeared

Use the narrowest probe:

- article RSS/listing sources: `v2_probe.py` / `v2_source_diag.py`;
- Executive Show: `v2_knbr_probe.py`;
- Chronicle/Mercury radar: `v2_radar_probe.py`;
- MLB schedule: `v2_schedule_probe.py`;
- The Athletic RSS: `v2_athletic_probe.py`;
- The Athletic metadata: `v2_athletic_meta_probe.py`.

A publisher/source failure should be fixed in its adapter or accepted as temporary degradation. Avoid changing unrelated adapters to accommodate it.

## Troubleshooting: wrong duplicate won

Check:

1. Are the headlines correctly in the same event cluster/event family?
2. Are they both `news`, or is one legitimately `analysis`?
3. Does the selected candidate have a real author/source prior?
4. Was comparable-source rotation applied? Inspect `rotation_applied`, `representatives`, alternatives and recent source counts.
5. Did one candidate have a 90+ minute early-reporting lead?
6. Did historical state suppress one role before comparison?
7. Is this actually game coverage, where root ordering follows a different rule?

Add a regression test for a real misclassification before broadening clustering or rotation.

## Troubleshooting: game thread looks wrong

Check `v2_game_threads.py`, `v2_mlb_schedule.py`, diagnostics and `state.json.game_threads`.

Important distinctions:

- schedule-grounded group: `schedule_grounded=true`, stable `game_pk`;
- schedule fallback: Pacific baseball-day + opponent;
- new thread: earliest published available core writer gets root;
- existing thread: root is immutable and new stories append;
- legacy migration: gamePk group can reuse an existing date/opponent thread;
- doubleheader: separate gamePk values should separate stories based on which game had most recently started.

If schedule matching is incorrect, do not delete a live Bluesky thread. Fix the matcher/test and preserve existing refs.

## Bluesky presentation failures

Current presentation is headline-first text + exact-hostname rich-text link + optional native image. External cards are intentionally not used.

- Articles: `Read at <hostname> →`
- KNBR Executive Show: `Listen at <hostname> →`
- Only the hostname is linked, avoiding Bluesky's misleading-link warning.
- Images are native `app.bsky.embed.images` with aspect ratio when available.
- Missing/blocked images should degrade to text + direct link, not suppress the story.

## Adding a new source

Before coding:

1. look for official team/category RSS;
2. look for an official sitemap or clean listing;
3. reuse a mature parser/package if cleanup is nontrivial;
4. only then consider a narrow source-specific adapter;
5. use targeted radar only when direct access is genuinely blocked and attribution can be tightly constrained.

For audio/podcast sources, prefer a dedicated show/playlist feed over broad station scraping and use an appropriate presentation verb such as `Listen at`.

## Safe-change checklist

Before merging:

- [ ] Production entrypoint remains `v2_bot.py`
- [ ] Direct destination URL is preserved
- [ ] No broad Google News production path was introduced
- [ ] One broken source cannot abort discovery
- [ ] Article-page enrichment failure remains nonblocking
- [ ] Same-event differentiated analysis is not accidentally collapsed
- [ ] Comparable rotation is not used as a broad quality override
- [ ] Schedule failure retains the game-grouping fallback
- [ ] Live game roots/legacy refs are preserved
- [ ] Dry run cannot post or mutate state
- [ ] Full V2 PR validation passed on the exact head commit
- [ ] No manual live Bluesky run was used as a test
- [ ] Documentation reflects product behavior
