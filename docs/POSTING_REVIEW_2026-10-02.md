# Giants News Bot: two-week posting review and offseason update

Reviewed September 18, 2026, 03:40 PM through October 02, 2026, 03:40 PM Pacific. Repository snapshot: `90de2d8baa5fab69843222044e6a74dcbc674709`. Public-feed snapshot retrieved directly through Bluesky's public author-feed API; repository state supplies posting metadata and production heartbeats.

## What was posted

- **97 posts visible in the public feed**, including 6 threaded replies and 91 top-level posts.
- **98 posting records in repository state:** 83 standalone stories and 15 game-story posts (game roots are top-level posts).
- The difference is a Brock Purdy/49ers story retained in state but absent from the public feed at review time. Its absence does not establish who removed it or when. The September 21 Mercury safety fix already addresses that class; do not erase history or repost it.
- **58 recorded production scans** in the window, all marked completed, with no recorded source failures. Ten selected neither standalone nor game stories. These heartbeats summarize adapter calls, not independent proof that every upstream query was healthy.
- Public engagement at snapshot: **14 likes and 6 reposts**. The small sample is insufficient to optimize policy around engagement alone.

Visible posts by publication (matched to state through direct article URLs):

| Publication | Posts |
| --- | ---: |
| NBC Sports Bay Area | 23 |
| San Francisco Chronicle | 17 |
| San Francisco Standard | 12 |
| Mercury News | 11 |
| The Athletic | 11 |
| MLB.com | 9 |
| SFGATE | 9 |
| FanGraphs | 4 |
| KNBR | 1 |

## Findings and changes

1. **FanGraphs relevance leakage.** [The White Sox ALDS recap](https://bsky.app/profile/giantsnewsbot.bsky.social/post/3mwsjmexjzc24) reached a Giants-only account on October 1. A team-category feed and targeted writer query can surface broad baseball pieces. Direct adapters, targeted radar and the selector now require explicit Giants/San Francisco evidence in the headline, article slug or first 600 characters of the structured summary. Mixed-team scouting remains eligible with explicit Giants evidence. Unknown relevance is rejected conservatively; this can miss a useful broad scouting item whose Giants section is buried deep in the article.
2. **Recurring chats slipped through.** The September 25 “Eric Longenhagen Prospects Chat: 9/25” post escaped a filter requiring a four-digit year. Both short-date and full-date chat headlines are now rejected.
3. **Repeated event reporting.** Camilo Doval's waiver claim appeared through NBC and The Athletic; Logan Webb's Willie Mac Award appeared through MLB.com, NBC and Chronicle. Normalize plural waivers/claim phrasing and the named award to improve clustering across headlines, while retaining the existing allowance for differentiated analysis. This does not retroactively delete posts.
4. **Feature starvation.** The selector counted raw high-quality injury headlines before checking whether they were stale or already posted. The latest heartbeats repeatedly showed two deferred features even on zero-selection runs. Deferral now counts fresh, unposted event clusters; repeated headlines from one event do not independently inflate the count.
5. **Season-review routing risk.** “What we learned” could label a season review as a game story, applying a 30-hour window or seeking an unrelated game. Explicit season-review/offseason subjects now take precedence in classification and routing. Actual final-game recaps still use the game lane.
6. **Offseason volume and order.** New policy prioritizes confirmed roster/contracts/coaching/injury news, then Giants development, pitching/payroll plans, roster deadlines and season analysis. Useful features remain eligible. Mock trades/trade proposals are excluded; attributed market reporting stays eligible with publisher qualifiers. No post quota is introduced.

## Offseason operating policy

- Explicit `SEASON_MODE=offseason` production default; repository variable can override it. Local default remains `inseason` for compatibility.
- **Weekdays:** 8:30 AM, 2:30 PM, 7:30 PM Pacific.
- **Weekends:** 8:30 AM, 1:30 PM, 5:30 PM Pacific.
- **Two standalone stories per run; six routine stories per Pacific calendar day.** Confirmed moves/injury news can bypass the routine daily budget, but remain subject to run/source/event caps. Manual trusted stories retain priority and their daily-budget exception; dedupe still applies.
- Keep 72-hour standalone freshness, 30-hour game freshness, source diversity, exact URL history, differentiated analysis and existing game refs.
- Set the repository variable `SEASON_MODE=inseason` when Giants game coverage resumes, including any October postseason participation. This restores the late-night checks and three-story run cap. A generic calendar cannot reliably decide whether the Giants are still playing.
- Polls are collection windows, not guaranteed post times; Actions can be delayed.

## Follow-up editorial priorities

Focus on arbitration/tender/option decisions, qualifying offers, Rule 5 and 40-man protection, original free-agent/trade reporting, pitching and bullpen construction, prospect development in the Fall League/winter ball, coaching/front-office decisions and substantive 2027 analysis. Preserve labor/lockout reporting when it materially affects the Giants or Oracle Park; avoid making general MLB playoff coverage a substitute for Giants news. Let quiet periods stay quiet.

Keep the current targeted source list. Do not add broad MLB feeds or generic rumor aggregation to fill space. KNBR's Executive Show can be seasonal; retain the existing Thursday morning check without assuming weekly offseason episodes.

## Verification

- 125 deterministic tests pass, including the actual Bash cadence gate under PDT/PST in both modes (32 combinations), Pacific-day budget boundaries, news-budget exceptions, actual leaked titles, season-review routing and full dry-run posting/state isolation.
- Live dry run: 148 candidates discovered, zero new posts selected against copied production state, no recorded adapter errors. Core-writer radar returned zero candidates, so this run does not demonstrate that Chronicle/Mercury radar discovery is working; CI diagnostics and future heartbeats remain useful.
- Copied state remained byte-for-byte unchanged; production `state.json` is excluded from the update.
- GitHub PR validation exercises live feeds, KNBR, schedule, targeted radar, production-state and clean-slate selection, and a full state-preserving offseason dry run.

## Implementation limits

Editorial priority uses headline cues rather than reading full articles. A vague confirmed transaction headline may not receive the news-budget exception; an ambiguous headline can be misclassified. Original publisher qualifiers are preserved. The daily budget relies on retained successful-post history and therefore survives normal scheduled runs. Calendar mode is deliberately operator-controlled; switch it back before game coverage resumes.
