import json
import tempfile
import unittest
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from feedparser import FeedParserDict

from config import Settings
from v2_bot import discover_articles, main
from v2_probe import discover_baggarly, make_article
from v2_selector import select_articles

NOW = datetime.now(timezone.utc)


def story(title="Giants sign veteran starter", source="The Athletic", author="Andrew Baggarly"):
    return {"title": title, "source": source, "author": author, "quality": "high",
            "url": "https://www.nytimes.com/athletic/99999/giants-veteran/",
            "published": NOW.isoformat(), "author_preference": "elite"}


def empty_state():
    return {"posted_urls": {}, "posted_stories": [], "game_threads": [], "run_history": []}


class FastAuthorTests(unittest.TestCase):
    def test_official_author_feed_supplies_missing_byline_and_rejects_other_teams(self):
        feed = FeedParserDict(feed=FeedParserDict(title="Andrew Baggarly - The Athletic"), entries=[
            FeedParserDict(title="Giants offseason pitching priorities", link="https://www.nytimes.com/athletic/123/giants/", published=NOW.isoformat()),
            FeedParserDict(title="Phillies win playoff game", link="https://www.nytimes.com/athletic/124/phillies/", published=NOW.isoformat()),
            FeedParserDict(title="Giants offseason report", link="https://other.example/athletic/125/", published=NOW.isoformat()),
        ])
        with patch("v2_probe.parse_feed", return_value=feed):
            articles = discover_baggarly()
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].author, "Andrew Baggarly")
        self.assertEqual(articles[0].quality, "high")

    def test_unexpected_author_feed_is_rejected(self):
        feed = FeedParserDict(feed=FeedParserDict(title="MLB - The Athletic"), entries=[])
        with patch("v2_probe.parse_feed", return_value=feed), self.assertRaises(RuntimeError):
            discover_baggarly()

    def test_fast_scan_only_calls_baggarly_and_does_not_apply_manual_input(self):
        item = make_article(source="The Athletic", title="Giants pitching outlook", url=story()["url"],
                            author="Andrew Baggarly", published=NOW.isoformat())
        with patch("v2_bot.discover_baggarly", return_value=[item]) as author_feed, \
             patch("v2_bot.DISCOVERERS", [lambda: self.fail("full discovery ran")]), \
             patch("v2_bot._manual_story_article") as manual:
            author_feed.__name__ = "discover_baggarly"
            articles, health = discover_articles(fast_lane=True)
        author_feed.assert_called_once()
        manual.assert_not_called()
        self.assertEqual(articles, [asdict(item)])
        self.assertTrue(health["baggarly"]["ok"])

    def test_fast_feed_failure_is_visible_as_failed_run(self):
        with patch("v2_bot.discover_baggarly", side_effect=RuntimeError("feed HTTP 503")) as author_feed:
            author_feed.__name__ = "discover_baggarly"
            with self.assertRaisesRegex(RuntimeError, "fast discovery failed"):
                discover_articles(fast_lane=True)

    def test_baggarly_does_not_use_the_six_story_routine_budget(self):
        state = empty_state()
        state["posted_stories"] = [{"title": f"Routine feature {i}", "source": "SFGATE",
                                    "posted_at": NOW.isoformat()} for i in range(6)]
        # A feature, not a transaction, must still receive the author exception.
        item = story("Giants offseason pitching priorities")
        result = select_articles([item], state, now=NOW, season_mode="offseason")
        self.assertEqual(len(result["selected"]), 1)
        state["posted_stories"].append({**item, "posted_at": NOW.isoformat()})
        result = select_articles([], state, now=NOW, season_mode="offseason")
        self.assertEqual(result["routine_posted_today"], 6)

    def test_earlier_outlet_does_not_suppress_baggarly_but_own_event_does(self):
        state = empty_state()
        earlier = {**story(source="NBC Sports Bay Area", author="Alex Pavlovic"),
                   "url": "https://www.nbcsportsbayarea.com/giants/veteran/", "posted_at": NOW.isoformat()}
        state["posted_stories"] = [earlier]
        result = select_articles([story()], state, now=NOW, season_mode="offseason")
        self.assertEqual(len(result["selected"]), 1)
        state["posted_stories"] = [{**earlier, "source": "The Athletic", "author": "Andrew Baggarly"}]
        result = select_articles([story()], state, now=NOW, season_mode="offseason")
        self.assertEqual(result["selected"], [])

    def test_same_url_is_never_reposted(self):
        item = story()
        state = empty_state()
        state["posted_urls"] = {item["url"]: NOW.isoformat()}
        self.assertEqual(select_articles([item], state, now=NOW, season_mode="offseason")["selected"], [])

    def test_quiet_production_fast_scan_does_not_write_state_or_heartbeat(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps(empty_state()), encoding="utf-8")
            before = path.read_bytes()
            settings = Settings(fast_lane=True, state_file=str(path))
            with patch("v2_bot.Settings", return_value=settings), \
                 patch("v2_bot.discover_articles", return_value=([], {"baggarly": {"ok": True, "count": 0}})), \
                 patch("v2_bot.save_state") as save, patch("v2_bot.bsky_login") as login:
                main()
            save.assert_not_called()
            login.assert_not_called()
            self.assertEqual(path.read_bytes(), before)

    def test_fast_dry_run_selects_new_article_without_posting_and_uses_six_hour_window(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps(empty_state()), encoding="utf-8")
            before = path.read_bytes()
            settings = Settings(fast_lane=True, dry_run=True, season_mode="offseason", state_file=str(path),
                                diagnostics_file=str(Path(directory) / "diagnostics.json"))
            item = story("Giants offseason pitching priorities")
            old = {**item, "url": item["url"] + "old", "published": (NOW - timedelta(hours=7)).isoformat()}
            with patch("v2_bot.Settings", return_value=settings), \
                 patch("v2_bot.discover_articles", return_value=([item, old], {"baggarly": {"ok": True, "count": 2}})), \
                 patch("v2_bot.enrich_card_metadata", return_value={}), \
                 patch("v2_bot.post_to_bluesky") as post, patch("v2_bot.bsky_login") as login:
                main()
            post.assert_not_called()
            login.assert_not_called()
            self.assertEqual(path.read_bytes(), before)
            d = json.loads(Path(settings.diagnostics_file).read_text(encoding="utf-8"))
            self.assertTrue(d["fast_lane"])
            self.assertEqual(d["selection"]["hours_back"], 6)
            self.assertEqual(len(d["selection"]["selected"]), 1)


if __name__ == "__main__":
    unittest.main()
