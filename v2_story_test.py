from datetime import datetime, timezone
import unittest

from v2_story import candidate_preference_key, same_story, story_role


class StoryDedupeTests(unittest.TestCase):
    def test_chapman_surgery_clusters(self):
        self.assertTrue(same_story(
            "Chapman looking toward 2027 after season-ending surgery",
            "Giants' Matt Chapman expects to have season-ending surgery for abdominal injury",
        ))

    def test_all_star_host_clusters(self):
        self.assertTrue(same_story(
            "Giants to host 2028 All-Star Game at Oracle Park",
            "San Francisco announced as host of 2028 MLB All-Star Game",
        ))

    def test_krukow_retirement_clusters(self):
        self.assertTrue(same_story(
            "Iconic Giants voice Mike Krukow announces retirement at end of season",
            "SF Giants announcer Mike Krukow to retire at end of season",
        ))

    def test_wilkinson_transaction_wording_clusters(self):
        brisbee = "The Giants promoted Matt 'Tugboat' Wilkinson, who has a chance to be a fan favorite"
        slusser = "Giants call up Matt ‘Tugboat’ Wilkinson with eye toward 2027 rotation options"
        mlb = "LHP Tugboat Wilkinson -- who throws an 'Invisiball' -- gets call from Giants (source)"
        self.assertTrue(same_story(brisbee, slusser))
        self.assertTrue(same_story(slusser, mlb))
        self.assertTrue(same_story(brisbee, mlb))

    def test_short_tugboat_promotion_headline_matches_callup_event(self):
        self.assertTrue(same_story(
            'Report: Giants promoting "Tugboat"',
            "Giants call up Matt ‘Tugboat’ Wilkinson with eye toward 2027 rotation options",
        ))
        self.assertTrue(same_story(
            'Report: Giants promoting "Tugboat"',
            "LHP Tugboat Wilkinson -- who throws an 'Invisiball' -- gets call from Giants (source)",
        ))

    def test_brisbee_wilkinson_is_analysis_while_transaction_beats_are_news(self):
        self.assertEqual(story_role({
            "source": "The Athletic",
            "author": "Grant Brisbee",
            "title": "The Giants promoted Matt 'Tugboat' Wilkinson, who has a chance to be a fan favorite",
        }), "analysis")
        self.assertEqual(story_role({
            "source": "San Francisco Chronicle",
            "author": "Susan Slusser",
            "title": "Giants call up Matt ‘Tugboat’ Wilkinson with eye toward 2027 rotation options",
        }), "news")
        self.assertEqual(story_role({
            "source": "MLB.com",
            "author": "Maria Guardado",
            "title": "LHP Tugboat Wilkinson -- who throws an 'Invisiball' -- gets call from Giants (source)",
        }), "news")

    def test_distinct_webb_analysis_does_not_cluster(self):
        self.assertFalse(same_story(
            "Giants have squandered Logan Webb's talents; an anonymous group changed that for a day",
            "Logan Webb continues to set example for young teammates as Giants turn to future",
        ))

    def test_same_player_different_event_does_not_cluster(self):
        self.assertFalse(same_story(
            "Matt Chapman wins another Gold Glove",
            "Matt Chapman expects to have season-ending surgery for abdominal injury",
        ))

    def test_author_signal_can_overcome_secondary_source(self):
        dt = datetime(2026, 8, 16, tzinfo=timezone.utc)
        alex_simon = {
            "author": "Alex Simon",
            "author_preference": "good",
            "source_preference": "secondary",
            "_published_dt": dt,
        }
        anonymous_primary = {
            "author": "",
            "author_preference": "",
            "source_preference": "",
            "_published_dt": dt,
        }
        self.assertGreater(candidate_preference_key(alex_simon), candidate_preference_key(anonymous_primary))


if __name__ == "__main__":
    unittest.main()
