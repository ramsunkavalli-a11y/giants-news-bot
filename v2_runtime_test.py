import json
import os
import tempfile
import unittest

from bsky_client import build_post_text, post_to_bluesky
from models import Candidate
from v2_bot import (
    _existing_thread_key,
    _set_thread_state,
    clean_card_summary,
    load_state,
    mark_posted,
)


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class FakeSession:
    def __init__(self):
        self.last_payload = None

    def post(self, url, **kwargs):
        self.last_payload = kwargs.get("json")
        return FakeResponse({
            "uri": "at://did:plc:test/app.bsky.feed.post/abc",
            "cid": "bafytest",
        })


class RuntimeSmokeTests(unittest.TestCase):
    def test_athletic_paywall_post_text(self):
        candidate = Candidate(
            source="The Athletic",
            url="https://example.com/story",
            author="Andrew Baggarly",
            access="paywalled",
        )
        self.assertEqual(
            build_post_text(candidate),
            "The Athletic ($) · Andrew Baggarly",
        )

    def test_free_source_post_text_unchanged(self):
        candidate = Candidate(
            source="MLB.com",
            url="https://example.com/story",
            author="Maria Guardado",
            access="free",
        )
        self.assertEqual(build_post_text(candidate), "MLB.com · Maria Guardado")

    def test_sf_chronicle_display_name_is_short(self):
        candidate = Candidate(
            source="San Francisco Chronicle",
            url="https://example.com/story",
            author="Shayna Rubin",
        )
        self.assertEqual(build_post_text(candidate), "SF Chronicle · Shayna Rubin")

    def test_game_recap_moves_headline_into_post_text(self):
        candidate = Candidate(
            source="San Francisco Chronicle",
            url="https://example.com/game-story",
            title="Giants’ Turner Hill delivers go-ahead RBI in major-league debut",
            author="Shayna Rubin",
            discovered_via="game_thread:Google News core-writer radar",
        )
        self.assertEqual(
            build_post_text(candidate),
            "Game recap · SF Chronicle · Shayna Rubin\n"
            "Giants’ Turner Hill delivers go-ahead RBI in major-league debut",
        )

    def test_game_recap_card_does_not_repeat_headline(self):
        session = FakeSession()
        candidate = Candidate(
            source="Mercury News",
            url="https://example.com/game-story",
            post_url="https://example.com/game-story",
            title="Webb’s strong outing, Gilbert’s big day lead SF Giants to win over Rockies",
            summary="A recap of the Giants win.",
            author="Justice delos Santos",
            discovered_via="game_thread:Google News core-writer radar",
        )
        post_to_bluesky(
            session,
            candidate,
            "https://bsky.social",
            "did:plc:test",
            "jwt",
            5,
        )
        record = session.last_payload["record"]
        self.assertIn(candidate.title, record["text"])
        external = record["embed"]["external"]
        self.assertEqual(external["title"], "Mercury News")
        self.assertEqual(external["description"], "")

    def test_missing_state_initializes_game_threads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "missing-state.json")
            state = load_state(path)
            self.assertEqual(state["posted_urls"], {})
            self.assertEqual(state["posted_stories"], [])
            self.assertEqual(state["game_threads"], {})
            self.assertEqual(set(state), {"posted_urls", "posted_stories", "game_threads"})

    def test_load_state_drops_retired_crawler_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "legacy-state.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "posted_urls": {"https://example.com/a": "2026-08-16T00:00:00+00:00"},
                        "posted_stories": [{"title": "Example"}],
                        "game_threads": {"game:2026-08-16:test": {"root": {}}},
                        "redirect_cache": {"legacy": "large"},
                        "meta_cache": {"legacy": "large"},
                    },
                    handle,
                )
            state = load_state(path)
            self.assertEqual(
                set(state),
                {"posted_urls", "posted_stories", "game_threads"},
            )
            self.assertIn("https://example.com/a", state["posted_urls"])
            self.assertEqual(state["posted_stories"][0]["title"], "Example")
            self.assertIn("game:2026-08-16:test", state["game_threads"])

    def test_mark_posted_stores_game_kind(self):
        state = {"posted_urls": {}, "posted_stories": []}
        article = {
            "title": "Giants beat Rockies",
            "url": "https://www.mlb.com/giants/news/example?utm_source=test",
            "source": "MLB.com",
            "author": "Maria Guardado",
        }
        mark_posted(state, article, kind="game_story", game_key="game:2026-08-15:rockies")
        self.assertEqual(len(state["posted_stories"]), 1)
        self.assertEqual(state["posted_stories"][0]["kind"], "game_story")
        self.assertEqual(state["posted_stories"][0]["game_key"], "game:2026-08-15:rockies")
        self.assertIn("https://www.mlb.com/giants/news/example", state["posted_urls"])

    def test_newsletter_promo_is_not_used_as_card_summary(self):
        promo = (
            "This story was excerpted from Maria Guardado’s Giants Beat newsletter. "
            "To read the full newsletter, click here. And subscribe to get it regularly."
        )
        self.assertEqual(clean_card_summary(promo, promo), "")

    def test_existing_unknown_thread_is_reused_when_opponent_becomes_known(self):
        state = {
            "game_threads": {
                "game:2026-08-15:unknown": {
                    "root": {"uri": "at://root", "cid": "rootcid"},
                    "parent": {"uri": "at://parent", "cid": "parentcid"},
                }
            }
        }
        thread = {
            "key": "game:2026-08-15:rockies",
            "game_day": "2026-08-15",
            "opponent": "rockies",
        }
        self.assertEqual(
            _existing_thread_key(state, thread),
            "game:2026-08-15:unknown",
        )

    def test_thread_state_stores_root_and_latest_parent(self):
        state = {"game_threads": {}}
        thread = {
            "key": "game:2026-08-15:rockies",
            "game_day": "2026-08-15",
            "opponent": "rockies",
        }
        root = {"uri": "at://root", "cid": "rootcid"}
        parent = {"uri": "at://reply", "cid": "replycid"}
        _set_thread_state(state, thread["key"], thread, root, parent)
        stored = state["game_threads"][thread["key"]]
        self.assertEqual(stored["root"], root)
        self.assertEqual(stored["parent"], parent)

    def test_bluesky_reply_payload_has_root_and_parent_refs(self):
        session = FakeSession()
        candidate = Candidate(
            source="MLB.com",
            url="https://example.com/game-story",
            title="Giants beat Rockies",
        )
        root = {"uri": "at://did/root", "cid": "rootcid"}
        parent = {"uri": "at://did/parent", "cid": "parentcid"}
        result = post_to_bluesky(
            session,
            candidate,
            "https://bsky.social",
            "did:plc:test",
            "jwt",
            5,
            reply_root=root,
            reply_parent=parent,
        )
        reply = session.last_payload["record"]["reply"]
        self.assertEqual(reply["root"], root)
        self.assertEqual(reply["parent"], parent)
        self.assertEqual(result["uri"], "at://did:plc:test/app.bsky.feed.post/abc")


if __name__ == "__main__":
    unittest.main()
