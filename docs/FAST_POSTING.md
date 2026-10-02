# Near-publication posting

## Andrew Baggarly

The bot checks the official [Andrew Baggarly author RSS](https://www.nytimes.com/athletic/rss/author/andrew-baggarly/) every 15 minutes, at minutes 7, 22, 37 and 52 of every hour, year-round. These scheduled checks use `FAST_LANE=1` and discover only Baggarly's feed. Normal curated scans still run at their existing times and also include his author feed.

The publisher's author-scoped feed establishes attribution. Its expected feed title is checked before using the byline. This avoids guessing authors from the Athletic Giants feed, which currently omits bylines, or relying on blocked article pages. Accept only HTTPS Athletic article URLs and Giants-relevant headlines/summaries; his national playoff reporting is excluded.

Fast checks consider the last six hours of stories. Normal scans retain the 72-hour window to recover after longer outages. Exact URLs and Baggarly's own same-event/role history prevent repeats. An earlier outlet's article does not suppress his original reporting. Game stories retain existing schedule-grounded thread behavior.

Baggarly stories do not consume the six-story daily routine allowance. Each check still observes run/source caps; a burst of several eligible stories can require successive checks. Quiet fast scans leave state and run history untouched; successful posts and full curated scans retain heartbeats. A failed fast discovery raises an error instead of silently succeeding. Fast and normal scans share one production concurrency group, preventing competing writes to posting state.

The intended detection delay is up to 15 minutes after a story becomes visible in RSS, plus runner startup and posting time. Feed publication can lag the article, and [GitHub scheduled jobs can be delayed or dropped](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule). This is best-effort frequent polling, not an instant-publishing guarantee.

## Other RSS options verified October 2, 2026

Only Baggarly is enabled for frequent polling in this change. The following live feeds returned articles and offer candidates for additional author-specific checks:

| Writers | RSS surface | Current use / caveat |
| --- | --- | --- |
| Grant Brisbee | [Athletic author RSS](https://www.nytimes.com/athletic/rss/author/grant-brisbee/) | Newly verified author feed; normal discovery uses the Athletic Giants feed. Giants relevance must still be checked. |
| Tim Kawakami | [Athletic author RSS](https://www.nytimes.com/athletic/rss/author/tim-kawakami/) | Newly verified; covers other sports, so strict Giants filtering is required. |
| Ken Rosenthal | [Athletic author RSS](https://www.nytimes.com/athletic/rss/author/ken-rosenthal/) | Newly verified; national reporting, only Giants-specific stories should enter the feed. |
| Alex Pavlovic | [NBC Giants RSS](https://www.nbcsportsbayarea.com/feed/?category_name=san-francisco-giants) | Newly verified feed with bylines/times. Production currently uses NBC listing pages; an RSS adapter could support frequent checks without listing-page enrichment. |
| Maria Guardado | [MLB Giants RSS](https://www.mlb.com/giants/feeds/news/rss.xml) | Already active; author filtering is required because the feed includes national/prospect/promotional material. |
| John Shea, Kerry Crowley | [SF Standard Giants RSS](https://sfstandard.com/tag/san-francisco-giants/feed/) | Already active; feed bylines allow precise author selection. |
| Alex Simon, Gabe Fernandez and other SFGATE writers | [SFGATE Giants RSS](https://www.sfgate.com/sports/feed/San-Francisco-Giants-RSS-Feed-428.php) | Already active; supports individual/co-bylines. Retain editorial-quality filters. |
| Eric Longenhagen and other FanGraphs writers | [FanGraphs Giants category RSS](https://blogs.fangraphs.com/category/teams/giants/feed/) | Already active, but does not include every Giants prospect piece. Explicit Giants evidence and chat/roundup rejection remain necessary. The tested Longenhagen author-feed URL returned zero entries and is not a verified alternative. |

Shayna Rubin, Susan Slusser, Justice delos Santos and other Chronicle/Mercury targets currently use Google News radar rather than direct publisher RSS. Do not promise similarly prompt discovery from that transport.

The strongest next choices are Pavlovic and Guardado for beat news, followed by Shea/Crowley and Brisbee. Add authors deliberately; the existence of an RSS feed does not authorize making every writer's work an automatic posting exception.

## Validation

`v2_fast_test.py` covers official author attribution, wrong-feed/domain/non-Giants rejection, isolated discovery, visible feed failures, daily-budget exemption, cross-publisher and exact-URL dedupe, quiet-state immutability, six-hour freshness and a complete dry run without login/posting. The actual workflow gate is tested under both Pacific offsets. CI also runs live Baggarly-only discovery against a copy of production state and verifies byte-for-byte immutability.
