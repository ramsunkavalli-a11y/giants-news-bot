from datetime import datetime, timedelta, timezone
import unittest

from v2_selector import select_articles


NOW = datetime(2026, 8, 17, 22, 0, tzinfo=timezone.utc)


def article(source, title, author, preference, minutes=0):
    return {
        "source": source,
        "title": title,
        "url": f"https://example.com/{source.lower().replace(' ', '-')}/{abs(hash(title))}",
        "published": (NOW - timedelta(minutes=minutes)).isoformat(),
        "author": author,
        "author_preference": preference,
        "source_preference": "",
        "quality": "high",
    }


class SelectorEditorialTests(unittest.TestCase):
    def setUp(self):
        self.slusser = article(
            "San Francisco Chronicle",
            "Giants call up Matt ‘Tugboat’ Wilkinson with eye toward 2027 rotation options",
            "Susan Slusser",
            "very_good",
            8,
        )
        self.guardado = article(
            "MLB.com",
            "LHP Tugboat Wilkinson -- who throws an 'Invisiball' -- gets call from Giants (source)",
            "Maria Guardado",
            "good",
            12,
        )
        self.brisbee = article(
            "The Athletic",
            "The Giants promoted Matt 'Tugboat' Wilkinson, who has a chance to be a fan favorite",
            "Grant Brisbee",
            "fine",
            5,
        )

    def test_same_event_keeps_one_news_version_plus_deeper_analysis(self):
        selection = select_articles(
            [self.slusser, self.guardado, self.brisbee],
            {"posted_urls": {}, "posted_stories": [], "game_threads": {}},
            max_posts=3,
            now=NOW,
        )
        selected = {(item["source"], item.get("story_role")) for item in selection["selected"]}
        self.assertEqual(selected, {
            ("San Francisco Chronicle", "news"),
            ("The Athletic", "analysis"),
        })
        self.assertEqual(selection["reasons"].get("story_duplicate"), 1)

    def test_comparable_transaction_news_rotates_toward_less_used_source(self):
        history = [
            {
                "title": f"Unrelated Chronicle story {index}",
                "url": f"https://www.sfchronicle.com/story-{index}",
                "source": "San Francisco Chronicle",
                "author": "Susan Slusser",
                "kind": "standalone",
                "posted_at": (NOW - timedelta(days=1, minutes=index)).isoformat(),
            }
            for index in range(5)
        ]
        selection = select_articles(
            [self.slusser, self.guardado, self.brisbee],
            {"posted_urls": {}, "posted_stories": history, "game_threads": {}},
            max_posts=3,
            now=NOW,
        )
        selected = {(item["source"], item.get("story_role")) for item in selection["selected"]}
        self.assertEqual(selected, {
            ("MLB.com", "news"),
            ("The Athletic", "analysis"),
        })
        self.assertEqual(selection["reasons"].get("comparable_story_rotation"), 1)

    def test_large_early_reporting_lead_overrides_rotation(self):
        early_guardado = dict(self.guardado)
        early_guardado["published"] = (NOW - timedelta(hours=3)).isoformat()
        late_slusser = dict(self.slusser)
        late_slusser["published"] = (NOW - timedelta(minutes=10)).isoformat()
        history = [{
            "title": f"Unrelated MLB story {index}",
            "url": f"https://www.mlb.com/giants/news/old-{index}",
            "source": "MLB.com",
            "author": "Maria Guardado",
            "kind": "standalone",
            "posted_at": (NOW - timedelta(days=1, minutes=index)).isoformat(),
        } for index in range(6)]
        selection = select_articles(
            [late_slusser, early_guardado],
            {"posted_urls": {}, "posted_stories": history, "game_threads": {}},
            max_posts=3,
            now=NOW,
        )
        self.assertEqual(len(selection["selected"]), 1)
        self.assertEqual(selection["selected"][0]["source"], "MLB.com")

    def test_posted_news_does_not_suppress_new_deeper_analysis(self):
        state = {
            "posted_urls": {},
            "posted_stories": [{
                "title": self.slusser["title"],
                "url": self.slusser["url"],
                "source": self.slusser["source"],
                "author": self.slusser["author"],
                "kind": "standalone",
                "posted_at": (NOW - timedelta(hours=1)).isoformat(),
            }],
            "game_threads": {},
        }
        selection = select_articles([self.brisbee], state, max_posts=3, now=NOW)
        self.assertEqual(len(selection["selected"]), 1)
        self.assertEqual(selection["selected"][0]["story_role"], "analysis")

    def test_posted_analysis_suppresses_same_event_analysis(self):
        state = {
            "posted_urls": {},
            "posted_stories": [{
                "title": self.brisbee["title"],
                "url": self.brisbee["url"],
                "source": self.brisbee["source"],
                "author": self.brisbee["author"],
                "kind": "standalone",
                "posted_at": (NOW - timedelta(hours=1)).isoformat(),
            }],
            "game_threads": {},
        }
        fangraphs = article(
            "FanGraphs",
            "Scouting Matt Tugboat Wilkinson after the Giants callup",
            "Other Analyst",
            "",
            0,
        )
        selection = select_articles([fangraphs], state, max_posts=3, now=NOW)
        self.assertEqual(selection["selected"], [])
        self.assertEqual(selection["reasons"].get("story_already_posted"), 1)

    def test_broad_all_mlb_ranking_is_rejected_at_selection_boundary(self):
        ranking = article(
            "MLB.com",
            "Ranking every MLB farm system, 1-30",
            "Sam Dykstra, Jim Callis and Jonathan Mayo",
            "",
            0,
        )
        selection = select_articles(
            [ranking],
            {"posted_urls": {}, "posted_stories": [], "game_threads": {}},
            max_posts=3,
            now=NOW,
        )
        self.assertEqual(selection["selected"], [])
        self.assertEqual(selection["reasons"].get("quality_low"), 1)

    def test_derivative_farm_ranking_rewrite_is_rejected_at_selection_boundary(self):
        ranking = article(
            "NBC Sports Bay Area",
            "Giants earn another top-five farm system ranking by prominent prospect outlet",
            "Vince Lontz",
            "",
            0,
        )
        selection = select_articles(
            [ranking],
            {"posted_urls": {}, "posted_stories": [], "game_threads": {}},
            max_posts=3,
            now=NOW,
        )
        self.assertEqual(selection["selected"], [])
        self.assertEqual(selection["reasons"].get("quality_low"), 1)

    def test_defers_human_interest_feature_during_multiple_breaking_updates(self):
        feature = article(
            "NBC Sports Bay Area",
            "Krukisms: Giants broadcaster Mike Krukow breaks down origins of his catchphrases",
            "Alex Pavlovic",
            "elite",
            0,
        )
        injuries = [
            article(
                "MLB.com",
                "Giants make flurry of moves after players land on IL",
                "Maria Guardado",
                "good",
                0,
            ),
            article(
                "San Francisco Chronicle",
                "Giants place key player on injured list",
                "Susan Slusser",
                "very_good",
                0,
            ),
        ]
        selection = select_articles(
            [feature, *injuries],
            {"posted_urls": {}, "posted_stories": [], "game_threads": {}},
            max_posts=3,
            now=NOW,
        )
        self.assertNotIn(feature["url"], {item["url"] for item in selection["selected"]})
        self.assertEqual(selection["reasons"].get("deferred_for_breaking_news"), 1)


if __name__ == "__main__":
    unittest.main()
