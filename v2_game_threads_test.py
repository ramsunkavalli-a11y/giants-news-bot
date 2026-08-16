import unittest

from v2_game_threads import (
    baseball_day,
    extract_opponent,
    game_thread_key,
    group_game_articles,
)


class GameThreadTests(unittest.TestCase):
    def test_next_morning_coverage_stays_with_previous_baseball_day(self):
        late = {"published": "2026-08-15T23:30:00-07:00"}
        morning = {"published": "2026-08-16T08:00:00-07:00"}
        self.assertEqual(baseball_day(late), "2026-08-15")
        self.assertEqual(baseball_day(morning), "2026-08-15")

    def test_afternoon_game_story_uses_same_calendar_day(self):
        article = {"published": "2026-08-15T16:30:00-07:00"}
        self.assertEqual(baseball_day(article), "2026-08-15")

    def test_extracts_opponent(self):
        article = {"title": "What we learned as Giants bats fall flat in loss to Rockies"}
        self.assertEqual(extract_opponent(article), "rockies")

    def test_unknown_story_merges_into_only_known_opponent_that_day(self):
        articles = [
            {
                "title": "Gilbert, Webb shine in win",
                "published": "2026-08-15T22:30:00-07:00",
            },
            {
                "title": "Giants beat Rockies behind Webb",
                "published": "2026-08-15T22:10:00-07:00",
            },
        ]
        groups = group_game_articles(articles)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["key"], game_thread_key("2026-08-15", "rockies"))
        self.assertEqual(len(groups[0]["articles"]), 2)

    def test_different_baseball_days_do_not_merge(self):
        articles = [
            {
                "title": "Giants beat Rockies",
                "published": "2026-08-15T22:30:00-07:00",
            },
            {
                "title": "Giants lose to Rockies",
                "published": "2026-08-16T22:30:00-07:00",
            },
        ]
        groups = group_game_articles(articles)
        self.assertEqual(len(groups), 2)


if __name__ == "__main__":
    unittest.main()
