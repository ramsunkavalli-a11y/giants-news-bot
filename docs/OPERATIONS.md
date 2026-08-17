# Operations

## Production workflow

Production is `.github/workflows/giants-news-bot.yml` on `main`.

The workflow:

1. evaluates the Pacific-time cadence gate
2. checks out `main`
3. installs Python 3.11 dependencies
4. runs `python v2_bot.py`
5. uploads `diagnostics.json`
6. commits `state.json` back to `main` only if it changed

Required GitHub Actions secrets:

- `BSKY_IDENTIFIER`
- `BSKY_APP_PASSWORD`

The default Bluesky PDS is `https://bsky.social` and can be overridden with `BSKY_PDS`.

## Schedule

All intended times below are **America/Los_Angeles** local time.

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

The earlier weekend windows better match day-game reporting while retaining an evening/postgame scan for late Saturday games.

### Why the cron file looks more complicated

GitHub Actions cron is UTC and does not accept a named timezone. The workflow therefore registers both PDT and PST UTC equivalents and then checks the current `America/Los_Angeles` offset before running the bot. The inactive duplicate schedule exits before checkout/install.

Do not simplify the workflow to one fixed UTC schedule unless an hour of seasonal drift is acceptable.

Scheduled Actions are not guaranteed to start on the exact minute. A run beginning several or even tens of minutes late is a scheduler-delay issue, not automatically a bot failure.

## Production environment

Current important values set by the workflow:

```text
HOURS_BACK=72
GAME_HOURS_BACK=30
MAX_POSTS_PER_RUN=3
DIAGNOSTICS_ENABLED=1
DIAGNOSTICS_FILE=diagnostics.json
```

Other environment-backed settings are in `config.py`.

## Safe local/test execution

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Full dry run:

```bash
DRY_RUN=1 \
DIAGNOSTICS_ENABLED=1 \
HOURS_BACK=72 \
GAME_HOURS_BACK=30 \
MAX_POSTS_PER_RUN=3 \
python v2_bot.py
```

Use an alternate state file when experimenting with state behavior:

```bash
STATE_FILE=/tmp/test-state.json DRY_RUN=1 python v2_bot.py
```

A dry run must not post to Bluesky or mutate state.

## Validation workflow

`.github/workflows/v2-structured-probe.yml` is the pre-merge safety suite for V2/maintenance changes.

It intentionally goes beyond unit tests because the project depends on live public feeds whose formats can drift.

The suite includes:

1. `v2_story_test.py`
2. `v2_game_threads_test.py`
3. `v2_radar_test.py`
4. `v2_runtime_test.py`
5. live structured-source probe
6. contained radar probe
7. copy of current production `state.json`
8. realistic selection simulation against production state
9. clean-slate duplicate-choice simulation
10. full V2 dry run against copied production state
11. explicit comparison that dry-run state is unchanged
12. Athletic RSS probe
13. Athletic structured-metadata probe
14. structured source diagnostics
15. artifact upload of probe outputs

A behavioral PR should not be merged merely because unit tests pass if the full probe workflow is failing.

## Deployment

There is no separate server deployment. Merging to `main` changes what the next scheduled GitHub Actions run executes.

Normal change process:

1. branch from current `main`
2. make a small coherent change
3. ensure the V2 validation workflow runs on the exact branch head
4. review failures/diagnostics
5. confirm production-state dry run is unchanged
6. merge PR
7. allow the next scheduled production run to exercise the code naturally

Do **not** manually dispatch production just to prove a merge worked. A manual dispatch is a real bot run and can post live stories.

## State management

`state.json` is committed production state. Treat it as data.

Important content:

- `posted_urls`: exact/canonical URL posting history
- `posted_stories`: recent story metadata used for historical story dedupe
- `game_threads`: live Bluesky root/latest-parent refs for game threads

### Never casually do these

- replace `state.json` with `{}`
- delete `posted_urls` to make tests pass
- delete `game_threads` while a thread may still receive replies
- use production state as a writable local test fixture
- commit state produced by a dry run

If the schema needs migration, preserve all dedupe history and Bluesky refs, validate the migration on a copy first, and make the migration explicit in code/PR notes.

## Diagnostics

Production uploads `diagnostics.json` as a GitHub Actions artifact for every real bot run that passes the cadence gate.

The V2 CI workflow uploads a broader `v2-structured-probe` artifact containing source, radar, selection, runtime, Athletic, and diagnostics outputs.

When debugging, diagnostics are preferable to guessing from the Bluesky feed because the absence of a post can mean several different things:

- workflow did not run yet
- cadence gate intentionally skipped the UTC duplicate
- no new candidate was discovered
- candidate was stale
- exact URL already posted
- event/story already posted
- candidate was low-value or not Giants-specific enough
- a stronger duplicate won
- source diversity suppressed the event after its winner was chosen
- candidate belongs to a game thread and was handled there

## Troubleshooting: no post appeared

### 1. Did the workflow actually run?

Check the latest `Giants News Bot` Actions runs. Compare the run's creation/start timestamp with the intended Pacific slot.

If no scheduled run exists yet, do not debug article selection. GitHub may simply be late.

### 2. Did the cadence gate allow it?

Because both PDT and PST cron equivalents exist, some scheduled workflow invocations are expected to stop at the gate. Inspect the `Pacific-time cadence gate` output for:

- `event_schedule`
- `pacific_offset`
- `run_bot`

Only `run_bot=true` is a real production scan.

### 3. Did `Run V2 bot` execute successfully?

If not, inspect the first exception. Discovery is source-isolated, so one source error normally should not fail the run; authentication/state-write problems are more likely to fail the whole job.

### 4. Inspect diagnostics

Look for discovered counts, selected standalone items, game-thread actions, and rejection/dedupe reasons.

### 5. Remember the caps and dedupe rules

A valid article can be intentionally absent because:

- another outlet's version won the same event cluster
- that winning publication was already used in the standalone run
- the article URL/story is in recent state
- the article is a game recap already represented/queued in a game thread

Do not loosen filters merely because one run was quiet.

## Troubleshooting: source disappeared

Use the source-specific/structured probes before editing global selection logic.

- Direct RSS/listing issue: `v2_probe.py` / `v2_source_diag.py`
- Chronicle/Mercury radar issue: `v2_radar_probe.py`
- The Athletic feed issue: `v2_athletic_probe.py`
- The Athletic page metadata issue: `v2_athletic_meta_probe.py`

A publisher failure should be fixed in its adapter or accepted as temporary degradation. Avoid changing unrelated sources to accommodate it.

## Troubleshooting: wrong duplicate won

Check:

1. Are the two headlines correctly in the same event cluster?
2. Does the higher-priority author/source metadata exist on the candidate?
3. Is a byline missing because enrichment failed?
4. Did historical state cause one candidate to be excluded before comparison?
5. Is this actually game coverage, where the core-writer earliest-publication root rule is intentionally different?

Add a regression test for any real misclassification before changing the clustering rule.

## Troubleshooting: game thread looks wrong

Check `v2_game_threads.py` and `state.json.game_threads`.

Important distinctions:

- **new thread:** earliest published eligible core writer gets root
- **existing thread:** root cannot be changed; new stories append
- **unknown opponent:** an existing unknown-opponent thread may be reused when opponent becomes known
- **doubleheader:** current day+opponent heuristic may be ambiguous

Do not delete a thread's state to force a different root on Bluesky.

## Bluesky presentation failures

If text posts but the card has no image or weak metadata, first determine whether the publisher blocks article-page fetching.

This is expected sometimes for Chronicle/Mercury. The posting path should degrade gracefully. A missing image is not grounds to suppress valid reporting.

If the external card repeats a game headline, check `bsky_client.py`; game posts intentionally move the headline to the post text and suppress card title/description.

## Adding a new source

Before coding:

1. Look for an official team/category RSS feed.
2. Look for an official sitemap or clean category/team listing.
3. Search for a mature public parser/package if cleanup is nontrivial.
4. Only then consider a narrow source-specific page adapter.
5. Use targeted search/radar only when direct access is genuinely blocked and attribution can be constrained tightly.

For a new adapter:

- return direct publisher URLs
- isolate exceptions
- attach reliable publication time/source/title
- enforce Giants relevance
- classify obvious low-value page types locally
- add diagnostics
- add regression tests if the source introduces a new behavior class

Avoid creating a generic broad web crawler.

## Safe-change checklist

Before merging:

- [ ] Production entrypoint remains `v2_bot.py`
- [ ] Direct publisher URL is preserved
- [ ] No broad Google News production path was introduced
- [ ] One broken source cannot abort discovery
- [ ] Article-page enrichment failure remains non-blocking
- [ ] Standalone duplicate winner is selected before source diversity
- [ ] Game root behavior is unchanged unless intentionally modified
- [ ] Dry run cannot post or mutate state
- [ ] Full V2 CI passed on the exact head commit
- [ ] No manual live Bluesky run was used as a test
- [ ] Documentation updated if product behavior changed
