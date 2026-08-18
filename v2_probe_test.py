import unittest

from v2_probe import classify


class ProbeEditorialClassificationTests(unittest.TestCase):
    def test_free_prospect_stream_promo_is_low_value(self):
        quality, reason, _ = classify(
            "MLB.com",
            "Watch top 50 prospect Level play at High-A for FREE on Thursday",
            "Brendan Samson",
        )
        self.assertEqual(quality, "low")
        self.assertEqual(reason, "promotional_content")

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


if __name__ == "__main__":
    unittest.main()
