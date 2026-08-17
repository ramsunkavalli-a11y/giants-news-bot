import unittest

from v2_game_threads import (
    baseball_day,
    extract_opponent,
    game_thread_key,
    group_game_articles,
    is_game_story,
    order_game_articles,
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

    def test_recognizes_game_story_even_if_quality_was_high(self):
        article = {
            "title": "Opener strategy pays off as Houser earns win for first time since May",
            "quality": "high",
            "quality_reason": "known_author:good",
        }
        self.assertTrue(is_game_story(article))

    def test_go_ahead_rbi_headline_is_game_coverage(self):
        article = {
            "title": "Giants’ Turner Hill delivers go-ahead RBI in major-league debut",
            "quality": "high",
            "author": "Shayna Rubin",
        }
        self.assertTrue(is_game_story(article))

    def test_structured_summary_can_identify_game_analysis(self):
        article = {
            "title": "Giants have squandered Logan Webb's talents; an anonymous group changed that for a day",
            "summary": "Webb posted another quality start and showed again that he is a standard setter for the Giants.",
            "quality": "high",
        }
        self.assertTrue(is_game_story(article))

    def test_does_not_turn_general_analysis_into_game_story(self):
        article = {
            "title": "Building the next good Giants bullpen will require more than spending money on it",
            "quality": "high",
        }
        self.assertFalse(is_game_story(article))

    def test_first_published_core_writer_gets_root_regardless_of_tier(self):
        articles = [
            {
                "author": "Andrew Baggarly",
                "author_preference": "elite",
                "published": "2026-08-15T20:10:00-07:00",
                "url": "https://example.com/baggarly",
            },
            {
                "author": "Maria Guardado",
                "author_preference": "good",
                "published": "2026-08-15T20:02:00-07:00",
                "url": "https://example.com/guardado",
            },
            {
                "author": "Alex Pavlovic",
                "author_preference": "elite",
                "published": "2026-08-15T20:05:00-07:00",
                "url": "https://example.com/pavlovic",
            },
        ]
        ordered = order_game_articles(articles)
        self.assertEqual(ordered[0]["author"], "Maria Guardado")

    def test_non_core_writer_cannot_take_root_when_core_writer_is_available(self):
        articles = [
            {
                "author": "Other Writer",
                "author_preference": "elite",
                "published": "2026-08-15T19:45:00-07:00",
                "url": "https://example.com/other",
            },
            {
                "author": "John Shea",
                "author_preference": "good",
                "published": "2026-08-15T20:00:00-07:00",
                "url": "https://example.com/shea",
            },
        ]
        ordered = order_game_articles(articles)
        self.assertEqual(ordered[0]["author"], "John Shea")

    def test_no_core_writer_keeps_existing_quality_fallback(self):
        articles = [
            {
                "author": "Unknown Writer",
                "author_preference": "",
                "published": "2026-08-15T19:45:00-07:00",
                "url": "https://example.com/unknown",
            },
            {
                "author": "Grant Brisbee",
                "author_preference": "fine",
                "published": "2026-08-15T20:00:00-07:00",
                "url": "https://example.com/brisbee",
            },
        ]
        ordered = order_game_articles(articles)
        self.assertEqual(ordered[0]["author"], "Grant Brisbee")

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

    def test_different_baseball_days_do_not_merge_without_schedule(self):
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

    def test_next_day_followup_maps_to_actual_previous_game(self):
        article = {
            "title": "Tony Vitello reveals message in team meeting after Giants' loss to Rockies",
            "published": "2026-08-17T14:45:00-07:00",
            "author": "Vince Lontz",
        }
        schedule = [{
            "game_pk": 900001,
            "official_date": "2026-08-16",
            "game_date": "2026-08-16T20:05:00Z",
            "opponent": "rockies",
        }]
        groups = group_game_articles([article], schedule_games=schedule)
        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0]["schedule_grounded"])
        self.assertEqual(groups[0]["game_pk"], 900001)
        self.assertEqual(groups[0]["game_day"], "2026-08-16")
        self.assertEqual(groups[0]["key"], "game:900001")

    def test_doubleheader_articles_map_to_most_recent_started_game(self):
        schedule = [
            {
                "game_pk": 910001,
                "official_date": "2026-09-05",
                "game_date": "2026-09-05T18:00:00Z",
                "opponent": "dodgers",
            },
            {
                "game_pk": 910002,
                "official_date": "2026-09-05",
                "game_date": "2026-09-05T23:00:00Z",
                "opponent": "dodgers",
            },
        ]
        between_games = {
            "title": "Giants beat Dodgers in opener",
            "published": "2026-09-05T21:30:00Z",
        }
        after_nightcap = {
            "title": "Giants lose to Dodgers in nightcap",
            "published": "2026-09-06T03:00:00Z",
        }
        groups = group_game_articles([between_games, after_nightcap], schedule_games=schedule)
        by_pk = {group["game_pk"]: group for group in groups}
        self.assertEqual(set(by_pk), {910001, 910002})
        self.assertEqual(by_pk[910001]["articles"][0]["title"], between_games["title"])
        self.assertEqual(by_pk[910002]["articles"][0]["title"], after_nightcap["title"])


if __name__ == "__main__":
    unittest.main()
