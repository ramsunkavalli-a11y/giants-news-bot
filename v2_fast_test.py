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
from v2_probe import discover_baggarly, discover_pavlovic, discover_guardado, make_article
from v2_editorial import is_priority_author
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

    def test_fast_scan_only_calls_selected_author_adapters_and_does_not_apply_manual_input(self):
        item = make_article(source="The Athletic", title="Giants pitching outlook", url=story()["url"],
                            author="Andrew Baggarly", published=NOW.isoformat())
        def discover_test_author():
            return [item]
        with patch("v2_bot.FAST_DISCOVERERS", [discover_test_author]), \
             patch("v2_bot.DISCOVERERS", [lambda: self.fail("full discovery ran")]), \
             patch("v2_bot._manual_story_article") as manual:
            articles, health = discover_articles(fast_lane=True)
        manual.assert_not_called()
        self.assertEqual(articles, [asdict(item)])
        self.assertTrue(health["test_author"]["ok"])

    def test_fast_feed_failure_is_visible_as_failed_run(self):
        def discover_test_failure():
            raise RuntimeError("feed HTTP 503")
        with patch("v2_bot.FAST_DISCOVERERS", [discover_test_failure]):
            with self.assertRaisesRegex(RuntimeError, "Hourly author discovery failed"):
                discover_articles(fast_lane=True)

    def test_baggarly_does_not_use_the_six_story_routine_budget(self):
        state = empty_state()
        state["posted_stories"] = [{"title": f"Routine feature {i}", "source": "SFGATE",
                                    "posted_at": NOW.isoformat()} for i in range(6)]
        # A feature, not a transaction, must still receive the author exception.
        for source, author in [("The Athletic", "Andrew Baggarly"),
                               ("NBC Sports Bay Area", "Alex Pavlovic"),
                               ("MLB.com", "Maria Guardado")]:
            with self.subTest(author=author):
                item = story("Giants offseason pitching priorities", source=source, author=author)
                item["url"] += author.replace(" ", "-")
                result = select_articles([item], state, now=NOW, season_mode="offseason", fast_lane=True)
                self.assertEqual(len(result["selected"]), 1)
                state["posted_stories"].append({**item, "posted_at": NOW.isoformat()})
        result = select_articles([], state, now=NOW, season_mode="offseason")
        self.assertEqual(result["routine_posted_today"], 6)

    def test_earlier_outlet_does_not_suppress_baggarly_but_own_event_does(self):
        state = empty_state()
        earlier = {**story(source="NBC Sports Bay Area", author="Another Writer"),
                   "url": "https://www.nbcsportsbayarea.com/giants/veteran/", "posted_at": NOW.isoformat()}
        state["posted_stories"] = [earlier]
        result = select_articles([story()], state, now=NOW, season_mode="offseason")
        self.assertEqual(len(result["selected"]), 1)
        state["posted_stories"] = [{**earlier, "source": "The Athletic", "author": "Andrew Baggarly"}]
        result = select_articles([story()], state, now=NOW, season_mode="offseason")
        self.assertEqual(result["selected"], [])

    def test_hourly_same_event_can_keep_all_three_followed_writers(self):
        items = [story(),
                 {**story(source="NBC Sports Bay Area", author="Alex Pavlovic"), "url": "https://www.nbcsportsbayarea.com/giants/123/"},
                 {**story(source="MLB.com", author="Maria Guardado"), "url": "https://www.mlb.com/giants/news/starter"}]
        result = select_articles(items, empty_state(), now=NOW, season_mode="offseason", fast_lane=True)
        self.assertEqual(len(result["selected"]), 3)
        self.assertEqual(result["max_posts"], 3)
        self.assertEqual({item["author"] for item in result["selected"]},
                         {"Andrew Baggarly", "Alex Pavlovic", "Maria Guardado"})
        state = empty_state()
        state["posted_stories"] = [{**items[0], "posted_at": NOW.isoformat()}]
        result = select_articles(items[1:], state, now=NOW, season_mode="offseason", fast_lane=True)
        self.assertEqual(len(result["selected"]), 2)

    def test_other_writers_are_excluded_from_hourly_selection(self):
        item = story(source="NBC Sports Bay Area", author="Another Writer")
        self.assertFalse(is_priority_author(item))
        result = select_articles([item], empty_state(), now=NOW, fast_lane=True)
        self.assertEqual(result["selected"], [])
        self.assertEqual(result["reasons"]["not_hourly_author"], 1)

    def test_nbc_hourly_feed_only_accepts_pavlovic_giants_articles(self):
        def entry(author, url, title="Giants pitching outlook"):
            return FeedParserDict(author=author, title=title, link=url, published=NOW.isoformat())
        feed = FeedParserDict(entries=[
            entry("Alex Pavlovic", "https://www.nbcsportsbayarea.com/mlb/san-francisco-giants/pitching/123/"),
            entry("Other Writer", "https://www.nbcsportsbayarea.com/mlb/san-francisco-giants/report/124/"),
            entry("Alex Pavlovic", "https://www.nbcsportsbayarea.com/nfl/san-francisco-49ers/report/125/"),
            entry("Alex Pavlovic", "https://www.nbcsportsbayarea.com/mlb/san-francisco-giants/video/report/126/"),
            entry("Alex Pavlovic", "https://other.example/mlb/san-francisco-giants/pitching/127/"),
        ])
        with patch("v2_probe.parse_feed", return_value=feed):
            articles = discover_pavlovic()
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].author, "Alex Pavlovic")
        self.assertEqual(articles[0].published, NOW.isoformat())

    def test_mlb_hourly_feed_only_accepts_guardado_with_known_feed_typo(self):
        feed = FeedParserDict(entries=[
            FeedParserDict(author=author, title="Giants pitching outlook", link=url, published=NOW.isoformat())
            for author, url in [
                ("Maria Guardado", "https://www.mlb.com/giants/news/pitching"),
                ("Maria Guarado", "https://www.mlb.com/giants/news/roster"),
                ("Sam Dykstra", "https://www.mlb.com/giants/news/prospect"),
                ("", "https://www.mlb.com/giants/news/unsigned"),
                ("Maria Guardado", "https://www.mlb.com/phillies/news/playoffs"),
                ("Maria Guardado", "https://other.example/giants/news/pitching"),
            ]])
        with patch("v2_probe.parse_feed", return_value=feed):
            articles = discover_guardado()
        self.assertEqual(len(articles), 2)
        self.assertTrue(all(item.author == "Maria Guardado" for item in articles))

    def test_one_failed_source_does_not_block_other_author_feeds(self):
        def discover_failed():
            raise RuntimeError("HTTP 503")
        def discover_healthy():
            return [make_article(source="MLB.com", author="Maria Guardado", title="Giants pitching outlook", url=story()["url"])]
        with patch("v2_bot.FAST_DISCOVERERS", [discover_failed, discover_healthy]):
            items, health = discover_articles(fast_lane=True)
        self.assertEqual(len(items), 1)
        self.assertFalse(health["failed"]["ok"])
        self.assertTrue(health["healthy"]["ok"])

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
