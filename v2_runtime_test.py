import os
import tempfile
import unittest

from bsky_client import build_post_text
from models import Candidate
from v2_bot import load_state, mark_posted


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

    def test_missing_state_uses_list_for_story_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "missing-state.json")
            state = load_state(path)
            self.assertEqual(state["posted_urls"], {})
            self.assertEqual(state["posted_stories"], [])

    def test_mark_posted_stores_title_and_url(self):
        state = {"posted_urls": {}, "posted_stories": []}
        article = {
            "title": "Giants to host 2028 All-Star Game at Oracle Park",
            "url": "https://www.mlb.com/giants/news/example?utm_source=test",
            "source": "MLB.com",
            "author": "Maria Guardado",
        }
        mark_posted(state, article)
        self.assertEqual(len(state["posted_stories"]), 1)
        self.assertIn("title", state["posted_stories"][0])
        self.assertIn(
            "https://www.mlb.com/giants/news/example",
            state["posted_urls"],
        )


if __name__ == "__main__":
    unittest.main()
