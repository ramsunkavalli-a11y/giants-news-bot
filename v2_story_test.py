from datetime import datetime, timezone
import unittest

from v2_story import candidate_preference_key, same_story


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
