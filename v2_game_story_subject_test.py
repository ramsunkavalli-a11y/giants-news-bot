import unittest

from v2_game_threads import is_game_story


class GameStorySubjectTests(unittest.TestCase):
    def test_mercury_news_blown_out_headline_is_game_story(self):
        article = {
            "source": "Mercury News",
            "title": "Whisenhunt’s troubles continue as SF Giants blown out by Guardians",
            "summary": "",
        }
        self.assertTrue(is_game_story(article))

    def test_reaction_headline_is_not_game_story(self):
        article = {
            "source": "SFGATE",
            "title": "Kruk and Kuip rip SF Giants players: 'Concentration is out the door'",
            "summary": "",
        }
        self.assertFalse(is_game_story(article))

    def test_reaction_headline_stays_out_even_when_it_mentions_the_result(self):
        article = {
            "source": "SFGATE",
            "title": "Kruk and Kuip rip SF Giants players after loss to Guardians",
            "summary": "The Giants were blown out by Cleveland before the broadcasters criticized the team's focus.",
        }
        self.assertFalse(is_game_story(article))

    def test_generic_result_in_summary_does_not_turn_commentary_into_recap(self):
        article = {
            "source": "Example",
            "title": "Posey faces questions about the Giants' direction",
            "summary": "The comments came after the Giants' loss to the Guardians.",
        }
        self.assertFalse(is_game_story(article))

    def test_on_field_analysis_summary_can_still_rescue_nonstandard_headline(self):
        article = {
            "source": "Example",
            "title": "Webb keeps setting the standard",
            "summary": "Webb delivered another quality start in the Giants' latest game.",
        }
        self.assertTrue(is_game_story(article))

    def test_season_outlook_is_not_a_game_story(self):
        article = {
            "source": "San Francisco Standard",
            "title": "A 100-loss season looms for the Giants. Here’s who’s left trying to stop it",
        }
        self.assertFalse(is_game_story(article))

    def test_official_mlb_recap_uses_explicit_result_in_url_slug(self):
        article = {
            "source": "MLB.com",
            "title": "Devers brings healing power to banged-up club with tape-measure tater",
            "url": "https://www.mlb.com/giants/news/rafael-devers-hits-3-run-homer-as-giants-beat-reds",
        }
        self.assertTrue(is_game_story(article))


if __name__ == "__main__":
    unittest.main()
