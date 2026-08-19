import unittest
from unittest.mock import patch

from v2_probe import classify, discover_sf_standard


class ProbeEditorialClassificationTests(unittest.TestCase):
    def test_free_prospect_stream_promo_is_low_value(self):
        quality, reason, _ = classify(
            "MLB.com",
            "Watch top 50 prospect Level play at High-A for FREE on Thursday",
            "Brendan Samson",
        )
        self.assertEqual(quality, "low")
        self.assertEqual(reason, "promotional_content")

    def test_mlb_maria_guardado_story_remains_eligible(self):
        quality, _, _ = classify(
            "MLB.com",
            "Chapman looking toward 2027 after season-ending surgery",
            "Maria Guardado",
        )
        self.assertEqual(quality, "high")

    def test_mlb_other_writer_is_low_value(self):
        quality, reason, _ = classify(
            "MLB.com",
            "Triple-A OF Davidson hits his way onto Prospect Team of the Week",
            "Ben Weinrib",
        )
        self.assertEqual(quality, "low")
        self.assertEqual(reason, "mlb_non_guardado")

    def test_unsigned_mlb_story_is_low_value(self):
        quality, reason, _ = classify(
            "MLB.com",
            "Giants announce roster move before series opener",
            "",
        )
        self.assertEqual(quality, "low")
        self.assertEqual(reason, "mlb_non_guardado")

    def test_nbc_broadcaster_opinion_repackaging_is_low_value(self):
        quality, reason, _ = classify(
            "NBC Sports Bay Area",
            "Mike Krukow believes Tony Vitello returning to Giants in 2027 is a ‘great thing’",
            "Dylan Grausz",
        )
        self.assertEqual(quality, "low")
        self.assertEqual(reason, "broadcaster_quote_repackaging")

    def test_broadcaster_news_is_not_suppressed(self):
        quality, _, _ = classify(
            "NBC Sports Bay Area",
            "Giants' Mike Krukow recalls emotional retirement conversation with Duane Kuiper",
            "Alex Pavlovic",
        )
        self.assertEqual(quality, "high")

    def test_front_office_opinion_is_not_caught_by_broadcaster_rule(self):
        quality, _, _ = classify(
            "NBC Sports Bay Area",
            "Buster Posey believes Tony Vitello can lead Giants back to contention",
            "Alex Pavlovic",
        )
        self.assertEqual(quality, "high")

    @patch("v2_probe.articles_from_feed")
    def test_sf_standard_uses_dedicated_giants_feed(self, articles_from_feed):
        articles_from_feed.return_value = []

        discover_sf_standard()

        articles_from_feed.assert_called_once_with(
            source="San Francisco Standard",
            feed_url="https://sfstandard.com/tag/san-francisco-giants/feed/",
            section="Giants tag RSS",
            access="free",
            limit=30,
            require_giants_relevance=False,
        )


if __name__ == "__main__":
    unittest.main()
