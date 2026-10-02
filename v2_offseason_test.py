import json
import os
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from config import Settings
from v2_bot import main
from v2_editorial import fangraphs_giants_evidence, is_confirmed_move
from v2_game_threads import is_game_story
from v2_probe import classify
from v2_radar import CORE_WRITER_RADAR_TARGETS, radar_story_allowed
from v2_selector import select_articles
from v2_story import same_story

NOW = datetime(2026, 10, 2, 22, tzinfo=timezone.utc)


def article(title, source="The Athletic", hours=1):
    return {"title": title, "source": source, "url": "https://example.com/" + title.replace(" ", "-"),
            "published": (NOW - timedelta(hours=hours)).isoformat(), "quality": "high"}


def state(stories=()):
    return {"posted_urls": {}, "posted_stories": list(stories), "game_threads": {}}


class OffseasonTests(unittest.TestCase):
    def select(self, articles, history=(), **kwargs):
        return select_articles(articles, state(history), now=NOW, season_mode="offseason", **kwargs)

    def test_confirmed_move_beats_newer_features_and_run_cap_is_two(self):
        move = article("Giants sign starter to two-year contract", "MLB.com", 5)
        result = self.select([article("Life in the broadcast booth"),
                              article("Giants rotation options for 2027", "NBC Sports Bay Area"), move])
        self.assertEqual(result["selected"][0]["title"], move["title"])
        self.assertEqual(len(result["selected"]), 2)

    def test_routine_daily_budget_preserves_confirmed_news(self):
        history = [{**article(f"Giants scouting report {i}"), "posted_at": NOW.isoformat(),
                    "kind": "standalone"} for i in range(6)]
        move = article("Giants claim reliever off waivers", "MLB.com")
        result = self.select([article("Giants offseason payroll questions"), move], history)
        self.assertEqual([x["title"] for x in result["selected"]], [move["title"]])
        self.assertEqual(result["reasons"]["offseason_daily_cap"], 1)

    def test_pacific_day_budget_crosses_utc_midnight(self):
        now = datetime(2026, 10, 3, 1, tzinfo=timezone.utc)  # Oct 2, 6 PM PDT
        prior = {**article("Giants scouting overview"), "posted_at": NOW.isoformat()}
        result = select_articles([article("Giants payroll priorities")], state([prior]),
                                 now=now, season_mode="offseason", offseason_daily_limit=1)
        self.assertEqual(result["selected"], [])
        prior["posted_at"] = "2026-10-02T06:59:00+00:00"  # Oct 1, 11:59 PM PDT
        result = select_articles([article("Giants payroll priorities")], state([prior]),
                                 now=now, season_mode="offseason", offseason_daily_limit=1)
        self.assertEqual(len(result["selected"]), 1)

    def test_confirmed_history_and_game_replies_do_not_use_routine_budget(self):
        history = [{**article("Giants sign a starter"), "posted_at": NOW.isoformat()},
                   {**article("Giants fall to Dodgers"), "posted_at": NOW.isoformat(), "kind": "game_story"}]
        result = self.select([article("Giants farm system outlook")], history, offseason_daily_limit=1)
        self.assertEqual(len(result["selected"]), 1)
        self.assertEqual(result["routine_posted_today"], 0)

    def test_manual_story_retains_priority_and_history_dedupe(self):
        manual = {**article("Giants broadcaster wins community award"), "_manual_priority": True}
        history = [{**article("Giants payroll report"), "posted_at": NOW.isoformat()}]
        result = self.select([manual], history, offseason_daily_limit=1)
        self.assertEqual(len(result["selected"]), 1)
        posted = state(history)
        posted["posted_urls"][manual["url"]] = NOW.isoformat()
        self.assertEqual(select_articles([manual], posted, now=NOW, season_mode="offseason")["selected"], [])

    def test_offseason_skips_mock_trades_but_keeps_attributed_market_reporting(self):
        result = self.select([article("Giants mock trade proposal for an ace"),
                              article("Giants free agent interest reported by beat writer", "MLB.com")])
        self.assertEqual(len(result["selected"]), 1)
        self.assertEqual(result["reasons"]["offseason_hypothetical"], 1)
        self.assertFalse(is_confirmed_move(article("Giants could sign a free agent")))

    def test_old_posted_injuries_cannot_defer_new_feature(self):
        injuries = [article("Giants injury update on Webb"), article("Giants place Eldridge on IL", "MLB.com")]
        posted = state([{**x, "posted_at": NOW.isoformat()} for x in injuries])
        posted["posted_urls"] = {x["url"]: NOW.isoformat() for x in injuries}
        feature = article("The origins of Giants broadcaster catchphrases", "SFGATE")
        result = select_articles([*injuries, feature], posted, now=NOW)
        self.assertEqual([x["title"] for x in result["selected"]], [feature["title"]])

    def test_two_new_injury_events_still_defer_feature_inseason(self):
        result = select_articles([article("Giants injury update on Webb"),
                                  article("Giants place Eldridge on IL", "MLB.com"),
                                  article("The origins of Giants broadcaster catchphrases", "SFGATE")],
                                 state(), now=NOW)
        self.assertEqual(len(result["selected"]), 2)
        self.assertEqual(result["reasons"]["deferred_for_breaking_news"], 1)

    def test_season_review_is_high_quality_and_not_a_game_even_with_old_label(self):
        title = "What we learned from the Giants' 2026 season"
        self.assertEqual(classify("NBC Sports Bay Area", title)[0], "high")
        self.assertFalse(is_game_story({"title": title, "quality_reason": "game_story_or_postgame_analysis"}))
        self.assertTrue(is_game_story(article("What we learned as Giants beat Dodgers")))

    def test_actual_fangraphs_leak_is_rejected_at_selector_and_radar(self):
        bad = article("Two Huge Innings Catapult White Sox Into ALDS", "FanGraphs")
        self.assertFalse(fangraphs_giants_evidence(bad))
        target = next(x for x in CORE_WRITER_RADAR_TARGETS if x.source == "FanGraphs")
        self.assertFalse(radar_story_allowed(target, bad["title"], bad["url"]))
        self.assertEqual(self.select([bad])["reasons"]["not_giants_focused"], 1)
        mixed = article("Arizona scouting notes", "FanGraphs")
        mixed["summary"] = "Scouting the Giants' young pitchers alongside Padres prospects."
        self.assertTrue(fangraphs_giants_evidence(mixed))
        self.assertEqual(len(self.select([mixed])["selected"]), 1)

    def test_short_date_prospects_chat_is_rejected(self):
        title = "Eric Longenhagen Prospects Chat: 9/25"
        self.assertEqual(classify("FanGraphs", title, "Eric Longenhagen")[0], "low")
        self.assertEqual(self.select([article(title, "FanGraphs")])["selected"], [])

    def test_observed_waiver_and_award_duplicates_match(self):
        self.assertTrue(same_story("Giants amazingly claim Camilo Doval off waivers after trading him last summer",
                                   "Giants claim former closer Camilo Doval off waivers"))
        self.assertTrue(same_story("Clubhouse leader Webb honored to win Willie Mac Award",
                                   "Logan Webb named Giants' 2026 Willie Mac Award winner despite disappointing year"))
        self.assertFalse(same_story("Logan Webb wins Willie Mac Award", "Logan Webb needs offseason surgery"))

    def test_invalid_mode_fails_before_discovery(self):
        with self.assertRaises(ValueError):
            Settings(season_mode="offseasn")

    def test_full_offseason_dry_run_preserves_state_and_never_logs_in(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps(state()), encoding="utf-8")
            before = path.read_bytes()
            settings = Settings(dry_run=True, season_mode="offseason", state_file=str(path),
                                diagnostics_file=str(Path(directory) / "diag.json"))
            with patch("v2_bot.Settings", return_value=settings), \
                 patch("v2_bot.discover_articles", return_value=([article("Giants sign starter")], {})), \
                 patch("v2_bot.enrich_card_metadata", return_value={}), \
                 patch("v2_bot.bsky_login") as login, patch("v2_bot.post_to_bluesky") as post:
                main()
            self.assertEqual(before, path.read_bytes())
            login.assert_not_called()
            post.assert_not_called()
            self.assertEqual(json.loads(Path(settings.diagnostics_file).read_text(encoding="utf-8"))
                             ["selection"]["season_mode"], "offseason")

    def test_actual_workflow_cadence_gate_for_both_modes_and_dst_offsets(self):
        windows_bash = Path("C:/Program Files/Git/bin/bash.exe")
        bash = str(windows_bash) if windows_bash.exists() else shutil.which("bash")
        if not bash:
            self.skipTest("Bash is required to exercise the GitHub runner's actual cadence gate")
        workflow = Path(__file__).with_name(".github") / "workflows" / "giants-news-bot.yml"
        block = workflow.read_text(encoding="utf-8").split("        run: |\n", 1)[1].split("\n      - name: Checkout", 1)[0]
        script = "\n".join(line[10:] for line in block.splitlines())
        cases = [
            ("-0700", "7 * * * *", True, True),
            ("-0800", "7 * * * *", True, True),
            ("-0700", "30 15 * * *", True, True),
            ("-0800", "30 16 * * *", True, True),
            ("-0700", "30 16 * * *", False, False),
            ("-0800", "30 15 * * *", False, False),
            ("-0700", "30 21 * * 1-5", True, True),
            ("-0800", "30 22 * * 1-5", True, True),
            ("-0700", "30 02 * * 2-6", True, True),
            ("-0800", "30 03 * * 2-6", True, True),
            ("-0700", "30 06 * * 2-6", True, False),
            ("-0800", "30 07 * * 2-6", True, False),
            ("-0700", "30 20 * * 6,0", True, True),
            ("-0800", "30 21 * * 6,0", True, True),
            ("-0700", "30 00 * * 0,1", True, True),
            ("-0800", "30 01 * * 0,1", True, True),
            ("-0700", "30 05 * * 0,1", True, False),
            ("-0800", "30 06 * * 0,1", True, False),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            for offset, schedule, inseason, offseason in cases:
                for mode, expected in [("inseason", inseason), ("offseason", offseason)]:
                    with self.subTest(offset=offset, schedule=schedule, mode=mode):
                        output.write_text("", encoding="utf-8")
                        body = script.replace("${{ github.event_name }}", "schedule").replace("${{ github.event.schedule }}", schedule)
                        result = subprocess.run([bash, "-c", f"date() {{ echo {offset}; }}\n" + body],
                                                env={**os.environ, "SEASON_MODE": mode, "GITHUB_OUTPUT": output.as_posix()},
                                                text=True, capture_output=True, timeout=10)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn(f"run_bot={str(expected).lower()}", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
