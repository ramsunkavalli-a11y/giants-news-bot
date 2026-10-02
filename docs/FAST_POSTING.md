# Near-publication posting

## Hourly beat-writer checks

The bot checks Andrew Baggarly, Alex Pavlovic and Maria Guardado once per hour, at minute 7 of every hour, year-round. These scheduled checks use `FAST_LANE=1` and discover only these three writers. Normal curated scans retain their existing schedule and publisher coverage.

| Writer | Hourly RSS | Attribution |
| --- | --- | --- |
| Andrew Baggarly | [Athletic author RSS](https://www.nytimes.com/athletic/rss/author/andrew-baggarly/) | Validated author-scoped feed title |
| Alex Pavlovic | [NBC Giants RSS](https://www.nbcsportsbayarea.com/feed/?category_name=san-francisco-giants) | Exact normalized article byline |
| Maria Guardado | [MLB Giants RSS](https://www.mlb.com/giants/feeds/news/rss.xml) | Exact normalized article byline, including the publisher's observed `Maria Guarado` typo |

Baggarly's author-scoped feed establishes attribution after checking its expected title. This avoids guessing authors from the Athletic Giants feed, which currently omits bylines, or relying on blocked article pages. His national playoff reporting is excluded. NBC and MLB entries require the named writer's byline and a valid HTTPS Giants article URL on the publisher's domain; other writers and NBC videos are excluded.

Hourly checks consider the last six hours of stories. Normal scans retain the 72-hour window to recover after longer outages. Exact URLs and each writer's own same-event/role history prevent repeats. Earlier coverage from another writer does not suppress their original reporting; all three can be selected when they cover the same event. Game stories retain existing schedule-grounded thread behavior.

All three writers bypass and do not consume the six-story daily routine allowance. An hourly check can select up to three standalone stories, one per writer; a burst from one writer can require successive checks. Regular offseason scans keep their two-story run cap. Quiet hourly scans leave state and run history untouched; successful posts and full curated scans retain heartbeats. A single failed feed does not prevent the other writers from being checked; diagnostics record each source's health, and failure of all three raises an error. Hourly and normal scans share one production concurrency group, preventing competing writes to posting state.

The intended detection delay is up to one hour after a story becomes visible in RSS, plus runner startup and posting time. Feed publication can lag the article, and [GitHub scheduled jobs can be delayed or dropped](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule). This is best-effort hourly polling, not an instant-publishing guarantee.

## Other RSS options verified October 2, 2026

Baggarly, Pavlovic and Guardado are enabled for hourly polling. The following additional live feeds returned articles and offer candidates for further author-specific checks:

| Writers | RSS surface | Current use / caveat |
| --- | --- | --- |
| Grant Brisbee | [Athletic author RSS](https://www.nytimes.com/athletic/rss/author/grant-brisbee/) | Newly verified author feed; normal discovery uses the Athletic Giants feed. Giants relevance must still be checked. |
| Tim Kawakami | [Athletic author RSS](https://www.nytimes.com/athletic/rss/author/tim-kawakami/) | Newly verified; covers other sports, so strict Giants filtering is required. |
| Ken Rosenthal | [Athletic author RSS](https://www.nytimes.com/athletic/rss/author/ken-rosenthal/) | Newly verified; national reporting, only Giants-specific stories should enter the feed. |
| John Shea, Kerry Crowley | [SF Standard Giants RSS](https://sfstandard.com/tag/san-francisco-giants/feed/) | Already active; feed bylines allow precise author selection. |
| Alex Simon, Gabe Fernandez and other SFGATE writers | [SFGATE Giants RSS](https://www.sfgate.com/sports/feed/San-Francisco-Giants-RSS-Feed-428.php) | Already active; supports individual/co-bylines. Retain editorial-quality filters. |
| Eric Longenhagen and other FanGraphs writers | [FanGraphs Giants category RSS](https://blogs.fangraphs.com/category/teams/giants/feed/) | Already active, but does not include every Giants prospect piece. Explicit Giants evidence and chat/roundup rejection remain necessary. The tested Longenhagen author-feed URL returned zero entries and is not a verified alternative. |

Shayna Rubin, Susan Slusser, Justice delos Santos and other Chronicle/Mercury targets currently use Google News radar rather than direct publisher RSS. Do not promise similarly prompt discovery from that transport.

The strongest additional choices are Shea/Crowley and Brisbee. Add authors deliberately; the existence of an RSS feed does not authorize making every writer's work an automatic posting exception.

## Validation

`v2_fast_test.py` covers all three feeds' attribution, wrong-feed/domain/non-Giants rejection, isolated discovery, partial and complete feed failures, daily-budget exemption, selection of all three writers on the same event, exact-URL and same-writer dedupe, quiet-state immutability, six-hour freshness and a complete dry run without login/posting. The actual hourly workflow gate is tested under both Pacific offsets. CI also runs live hourly-author discovery against a copy of production state and verifies byte-for-byte immutability.
