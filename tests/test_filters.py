from config import Settings
from filters import canonicalize_url, is_relevant_giants, is_story_url


def test_canonicalize_strips_tracking():
    settings = Settings()
    url = "https://example.com/path/story?utm_source=x&fbclid=123&id=9"
    assert canonicalize_url(url, settings) == "https://example.com/path/story?id=9"


def test_story_url_detection():
    assert is_story_url("https://www.sfgate.com/giants/article/sf-giants-win-series-finale-12345678.php")
    assert not is_story_url("https://www.sfgate.com/giants/")


def test_giants_relevance_disambiguates_nfl():
    assert is_relevant_giants("San Francisco Giants rally", "Big MLB comeback", [], "https://x.test/a")
    assert not is_relevant_giants("Giants sign quarterback", "New York Giants NFL update", [], "https://x.test/b")
