import unittest

from bsky_client import build_link_post
from models import Candidate
from v2_knbr import _giants_episode, _rss_url_from_playlist


class KnbrExecutiveShowTests(unittest.TestCase):
    def test_playlist_page_rss_link_is_discovered(self):
        html = '''
        <html><head>
          <link rel="alternate" type="application/rss+xml" href="https://www.omnycontent.com/d/playlist/example.rss" />
        </head></html>
        '''
        self.assertEqual(
            _rss_url_from_playlist(html),
            "https://www.omnycontent.com/d/playlist/example.rss",
        )

    def test_giants_executive_episode_is_accepted(self):
        self.assertTrue(_giants_episode(
            "Buster Posey discusses the Giants' young pitching",
            "The Giants president of baseball operations joins the Executive Show.",
        ))
        self.assertTrue(_giants_episode(
            "Zack Minasian on the trade deadline",
            "Giants GM Zack Minasian joins KNBR.",
        ))

    def test_49ers_only_executive_episode_is_rejected(self):
        self.assertFalse(_giants_episode(
            "John Lynch on the 49ers roster",
            "The 49ers president of football operations joins KNBR.",
        ))

    def test_knbr_audio_uses_listen_at_exact_hostname(self):
        candidate = Candidate(
            source="KNBR",
            url="https://omny.fm/shows/the-executives-show/buster-posey-giants",
            title="Buster Posey discusses the Giants' young pitching",
            author="The Executive Show",
        )
        text, facets = build_link_post(candidate)
        self.assertEqual(
            text,
            "Buster Posey discusses the Giants' young pitching\n"
            "KNBR · The Executive Show\n"
            "Listen at omny.fm →",
        )
        facet = facets[0]
        self.assertEqual(facet["features"][0]["uri"], candidate.url)
        encoded = text.encode("utf-8")
        linked = encoded[facet["index"]["byteStart"]:facet["index"]["byteEnd"]].decode("utf-8")
        self.assertEqual(linked, "omny.fm")


if __name__ == "__main__":
    unittest.main()
