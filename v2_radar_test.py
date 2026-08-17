import unittest
from unittest.mock import patch

from v2_probe import make_article
from v2_radar import (
    CORE_WRITER_RADAR_TARGETS,
    RadarTarget,
    discover_core_writer_radar,
    domain_matches,
    metadata_byline_includes_target,
    strip_google_source_suffix,
    unique_author_records,
)


class CoreWriterRadarTests(unittest.TestCase):
    def test_domain_match_requires_target_publisher(self):
        self.assertTrue(domain_matches("https://www.sfchronicle.com/sports/giants/a", "sfchronicle.com"))
        self.assertTrue(domain_matches("https://mercurynews.com/2026/a", "mercurynews.com"))
        self.assertFalse(domain_matches("https://example.com/giants", "sfchronicle.com"))

    def test_google_source_suffix_is_removed(self):
        self.assertEqual(
            strip_google_source_suffix(
                "Giants win again - San Francisco Chronicle",
                "San Francisco Chronicle",
            ),
            "Giants win again",
        )

    def test_same_url_matching_multiple_author_queries_is_rejected(self):
        slusser = RadarTarget("Susan Slusser", "San Francisco Chronicle", "sfchronicle.com")
        rubin = RadarTarget("Shayna Rubin", "San Francisco Chronicle", "sfchronicle.com")
        records = [
            {"target": slusser, "url": "https://www.sfchronicle.com/a"},
            {"target": rubin, "url": "https://www.sfchronicle.com/a"},
        ]
        self.assertEqual(unique_author_records(records), [])

    def test_target_writer_can_be_one_member_of_metadata_cobyline(self):
        self.assertTrue(metadata_byline_includes_target("Susan Slusser, J.D. Morris", "Susan Slusser"))
        self.assertTrue(metadata_byline_includes_target("J.D. Morris and Susan Slusser", "Susan Slusser"))
        self.assertFalse(metadata_byline_includes_target("J.D. Morris, John Shea", "Susan Slusser"))

    @patch("v2_radar.structured_meta_author", return_value="")
    @patch("v2_radar._feed_records")
    def test_unique_exact_author_query_can_supply_blocked_byline(self, feed_records, _meta):
        rubin = next(target for target in CORE_WRITER_RADAR_TARGETS if target.author == "Shayna Rubin")

        def records(target, _hours_back):
            if target != rubin:
                return []
            return [{
                "target": rubin,
                "url": "https://www.sfchronicle.com/sports/giants/test.html",
                "title": "Giants win behind late rally",
                "summary": "",
                "published": "Sun, 16 Aug 2026 02:07:47 GMT",
            }]

        feed_records.side_effect = records
        articles = discover_core_writer_radar()
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].author, "Shayna Rubin")
        self.assertEqual(articles[0].author_preference, "elite")
        self.assertEqual(articles[0].source, "San Francisco Chronicle")

    @patch("v2_radar.structured_meta_author", return_value="Susan Slusser, J.D. Morris")
    @patch("v2_radar._feed_records")
    def test_targeted_core_writer_cobyline_is_accepted(self, feed_records, _meta):
        slusser = next(target for target in CORE_WRITER_RADAR_TARGETS if target.author == "Susan Slusser")

        def records(target, _hours_back):
            if target != slusser:
                return []
            return [{
                "target": slusser,
                "url": "https://www.sfchronicle.com/sports/giants/all-star.html",
                "title": "Giants’ Oracle Park to host 2028 All-Star Game",
                "summary": "",
                "published": "Fri, 14 Aug 2026 16:00:00 GMT",
            }]

        feed_records.side_effect = records
        articles = discover_core_writer_radar()
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].author, "Susan Slusser")

    @patch("v2_radar.structured_meta_author", return_value="Different Writer")
    @patch("v2_radar._feed_records")
    def test_contradictory_metadata_vetoes_query_attribution(self, feed_records, _meta):
        target = CORE_WRITER_RADAR_TARGETS[0]
        feed_records.return_value = [{
            "target": target,
            "url": "https://www.sfchronicle.com/sports/giants/test.html",
            "title": "Giants story",
            "summary": "",
            "published": "Sun, 16 Aug 2026 02:07:47 GMT",
        }]
        self.assertEqual(discover_core_writer_radar(), [])

    def test_discoverer_failure_is_isolated_by_v2_runtime(self):
        import v2_bot

        def good_discoverer():
            return [make_article(
                source="MLB.com",
                title="Giants roster move",
                url="https://www.mlb.com/giants/news/test",
                published="Sun, 16 Aug 2026 02:07:47 GMT",
                author="Maria Guardado",
                access="free",
            )]

        def broken_radar():
            raise RuntimeError("radar unavailable")

        with patch.object(v2_bot, "DISCOVERERS", [good_discoverer, broken_radar]):
            articles, health = v2_bot.discover_articles()

        self.assertEqual(len(articles), 1)
        self.assertTrue(health["good_discoverer"]["ok"])
        self.assertFalse(health["broken_radar"]["ok"])


if __name__ == "__main__":
    unittest.main()
