import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone

from PIL import Image

from bsky_client import build_post_text, post_to_bluesky
from models import Candidate
from v2_bot import (
    _existing_thread_key,
    _set_thread_state,
    clean_card_summary,
    load_state,
    mark_posted,
)
from v2_selector import select_articles


class FakeResponse:
    def __init__(self, data=None, *, status_code=200, headers=None, raw=None):
        self._data = data or {}
        self.status_code = status_code
        self.headers = headers or {}
        self.raw = raw

    def raise_for_status(self):
        return None

    def json(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def jpeg_bytes(width=1200, height=675):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeSession:
    def __init__(self, *, image_available=False, image_bytes=None):
        self.last_payload = None
        self.image_available = image_available
        self.image_bytes = image_bytes if image_bytes is not None else jpeg_bytes()

    def get(self, url, **kwargs):
        if not self.image_available:
            return FakeResponse(status_code=404, raw=io.BytesIO(b""))
        return FakeResponse(
            status_code=200,
            headers={"Content-Type": "image/jpeg"},
            raw=io.BytesIO(self.image_bytes),
        )

    def post(self, url, **kwargs):
        if url.endswith("/xrpc/com.atproto.repo.uploadBlob"):
            return FakeResponse({"blob": {"$type": "blob", "ref": {"$link": "bafyimage"}}})
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

    def test_standalone_headline_is_first_in_post_text(self):
        candidate = Candidate(
            source="Mercury News",
            url="https://example.com/story",
            title="Despite lingering back issue, SF Giants’ Adames hopes to avoid IL stint",
            author="Justice delos Santos",
            access="free",
        )
        self.assertEqual(
            build_post_text(candidate),
            "Despite lingering back issue, SF Giants’ Adames hopes to avoid IL stint\n"
            "Mercury News · Justice delos Santos",
        )

    def test_sf_chronicle_display_name_is_short(self):
        candidate = Candidate(
            source="San Francisco Chronicle",
            url="https://example.com/story",
            author="Shayna Rubin",
        )
        self.assertEqual(build_post_text(candidate), "SF Chronicle · Shayna Rubin")

    def test_game_recap_headline_is_first_in_post_text(self):
        candidate = Candidate(
            source="San Francisco Chronicle",
            url="https://example.com/game-story",
            title="Giants’ Turner Hill delivers go-ahead RBI in major-league debut",
            author="Shayna Rubin",
            discovered_via="game_thread:Google News core-writer radar",
        )
        self.assertEqual(
            build_post_text(candidate),
            "Giants’ Turner Hill delivers go-ahead RBI in major-league debut\n"
            "Game recap · SF Chronicle · Shayna Rubin",
        )

    def test_no_image_game_recap_uses_matching_domain_link(self):
        session = FakeSession()
        candidate = Candidate(
            source="San Francisco Chronicle",
            url="https://www.sfchronicle.com/sports/giants/article/slusser-game-story-123.php",
            title="Giants’ Tony Vitello says play in loss to Rockies had feel of ‘junior college game’",
            author="Susan Slusser",
            discovered_via="game_thread:Google News core-writer radar",
        )
        post_to_bluesky(session, candidate, "https://bsky.social", "did:plc:test", "jwt", 5)
        record = session.last_payload["record"]
        self.assertNotIn("embed", record)
        self.assertEqual(
            record["text"],
            "Giants’ Tony Vitello says play in loss to Rockies had feel of ‘junior college game’\n"
            "Game recap · SF Chronicle · Susan Slusser\n"
            "Read at www.sfchronicle.com →",
        )
        facet = record["facets"][0]
        self.assertEqual(facet["features"][0]["uri"], candidate.url)
        encoded = record["text"].encode("utf-8")
        linked = encoded[facet["index"]["byteStart"]:facet["index"]["byteEnd"]].decode("utf-8")
        self.assertEqual(linked, "www.sfchronicle.com")

    def test_no_image_standalone_uses_matching_domain_link(self):
        session = FakeSession()
        candidate = Candidate(
            source="Mercury News",
            url="https://www.mercurynews.com/2026/08/17/example-story/",
            title="Despite lingering back issue, SF Giants’ Adames hopes to avoid IL stint",
            author="Justice delos Santos",
        )
        post_to_bluesky(session, candidate, "https://bsky.social", "did:plc:test", "jwt", 5)
        record = session.last_payload["record"]
        self.assertNotIn("embed", record)
        self.assertTrue(record["text"].startswith(
            "Despite lingering back issue, SF Giants’ Adames hopes to avoid IL stint\n"
            "Mercury News · Justice delos Santos\n"
        ))
        self.assertTrue(record["text"].endswith("Read at www.mercurynews.com →"))
        facet = record["facets"][0]
        self.assertEqual(facet["features"][0]["uri"], candidate.url)
        encoded = record["text"].encode("utf-8")
        linked = encoded[facet["index"]["byteStart"]:facet["index"]["byteEnd"]].decode("utf-8")
        self.assertEqual(linked, "www.mercurynews.com")

    def test_image_story_uses_native_image_with_aspect_ratio_and_matching_domain_link(self):
        session = FakeSession(image_available=True, image_bytes=jpeg_bytes(1200, 675))
        candidate = Candidate(
            source="NBC Sports Bay Area",
            url="https://www.nbcsportsbayarea.com/mlb/san-francisco-giants/example/1957101/",
            title="Giants' Landen Roupp looking for mechanical tweak",
            author="Taylor Wirth",
            image_url="https://example.com/image.jpg",
        )
        post_to_bluesky(session, candidate, "https://bsky.social", "did:plc:test", "jwt", 5)
        record = session.last_payload["record"]
        self.assertTrue(record["text"].startswith(
            "Giants' Landen Roupp looking for mechanical tweak\n"
            "NBC Sports Bay Area · Taylor Wirth\n"
        ))
        self.assertTrue(record["text"].endswith("Read at www.nbcsportsbayarea.com →"))
        facet = record["facets"][0]
        self.assertEqual(facet["features"][0]["uri"], candidate.url)
        encoded = record["text"].encode("utf-8")
        linked = encoded[facet["index"]["byteStart"]:facet["index"]["byteEnd"]].decode("utf-8")
        self.assertEqual(linked, "www.nbcsportsbayarea.com")
        self.assertEqual(record["embed"]["$type"], "app.bsky.embed.images")
        image = record["embed"]["images"][0]
        self.assertEqual(image["alt"], candidate.title)
        self.assertEqual(image["image"]["ref"]["$link"], "bafyimage")
        self.assertEqual(image["aspectRatio"], {"width": 1200, "height": 675})
        self.assertNotIn("external", record["embed"])

    def test_oversize_image_is_not_silently_truncated(self):
        session = FakeSession(image_available=True, image_bytes=b"x" * 2_000_001)
        candidate = Candidate(
            source="MLB.com",
            url="https://www.mlb.com/giants/news/example",
            title="Example Giants story",
            image_url="https://example.com/huge.jpg",
        )
        post_to_bluesky(session, candidate, "https://bsky.social", "did:plc:test", "jwt", 5)
        record = session.last_payload["record"]
        self.assertNotIn("embed", record)
        self.assertTrue(record["text"].endswith("Read at www.mlb.com →"))

    def test_invalid_image_bytes_degrade_to_text_link(self):
        session = FakeSession(image_available=True, image_bytes=b"not-a-real-image")
        candidate = Candidate(
            source="MLB.com",
            url="https://www.mlb.com/giants/news/example",
            title="Example Giants story",
            image_url="https://example.com/bad.jpg",
        )
        post_to_bluesky(session, candidate, "https://bsky.social", "did:plc:test", "jwt", 5)
        record = session.last_payload["record"]
        self.assertNotIn("embed", record)
        self.assertTrue(record["text"].endswith("Read at www.mlb.com →"))

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
            self.assertEqual(set(state), {"posted_urls", "posted_stories", "game_threads"})
            self.assertIn("https://example.com/a", state["posted_urls"])
            self.assertEqual(state["posted_stories"][0]["title"], "Example")
            self.assertIn("game:2026-08-16:test", state["game_threads"])

    def test_highlight_page_rejected_at_selection_boundary(self):
        now = datetime(2026, 8, 16, 18, 0, tzinfo=timezone.utc)
        article = {
            "source": "MLB.com",
            "title": "Rockies-Giants highlights",
            "url": "https://www.mlb.com/stories/game/823182",
            "published": now.isoformat(),
            "quality": "high",
        }
        selection = select_articles(
            [article],
            {"posted_urls": {}, "posted_stories": [], "game_threads": {}},
            now=now,
        )
        self.assertEqual(selection["selected"], [])
        self.assertEqual(selection["reasons"].get("quality_low"), 1)

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
        self.assertEqual(_existing_thread_key(state, thread), "game:2026-08-15:unknown")

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
