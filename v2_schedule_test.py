from datetime import date
import unittest

from v2_bot import _existing_thread_key
from v2_mlb_schedule import fetch_giants_schedule


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "dates": [{
                "date": "2026-08-16",
                "games": [{
                    "gamePk": 900001,
                    "officialDate": "2026-08-16",
                    "gameDate": "2026-08-16T20:05:00Z",
                    "status": {
                        "abstractGameState": "Final",
                        "detailedState": "Final",
                    },
                    "teams": {
                        "away": {"team": {"id": 115, "name": "Colorado Rockies"}},
                        "home": {"team": {"id": 137, "name": "San Francisco Giants"}},
                    },
                }],
            }],
        }


class FakeSession:
    def __init__(self):
        self.last_params = None

    def get(self, _url, **kwargs):
        self.last_params = kwargs.get("params")
        return FakeResponse()


class ScheduleTests(unittest.TestCase):
    def test_statsapi_schedule_maps_giants_opponent_and_game_pk(self):
        session = FakeSession()
        games = fetch_giants_schedule(
            date(2026, 8, 16),
            date(2026, 8, 17),
            session=session,
        )
        self.assertEqual(session.last_params["teamId"], 137)
        self.assertEqual(len(games), 1)
        self.assertEqual(games[0]["game_pk"], 900001)
        self.assertEqual(games[0]["official_date"], "2026-08-16")
        self.assertEqual(games[0]["opponent"], "rockies")

    def test_new_game_pk_key_reuses_existing_legacy_thread(self):
        state = {
            "game_threads": {
                "game:2026-08-16:rockies": {
                    "root": {"uri": "at://root", "cid": "rootcid"},
                    "parent": {"uri": "at://parent", "cid": "parentcid"},
                }
            }
        }
        thread = {
            "key": "game:900001",
            "game_pk": 900001,
            "game_day": "2026-08-16",
            "opponent": "rockies",
        }
        self.assertEqual(
            _existing_thread_key(state, thread),
            "game:2026-08-16:rockies",
        )


if __name__ == "__main__":
    unittest.main()
