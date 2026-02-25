from config import Settings
from filters import (
    canonicalize_url,
    extract_publisher_url_from_google_wrapper,
    is_relevant_giants,
    is_story_url,
    looks_like_google_wrapper,
    mlb_article_url_allowed,
    source_policy_allows,
    source_url_allowed,
)


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


def test_mlb_article_url_rules():
    assert mlb_article_url_allowed("https://www.mlb.com/giants/news/buster-posey-discusses-rotation-depth-at-camp")
    assert not mlb_article_url_allowed("https://www.mlb.com/giants/stats")
    assert not mlb_article_url_allowed("https://www.mlb.com/news/leaguewide-thing")


def test_source_rules_for_mlb_source():
    assert source_url_allowed("MLB Giants", "https://www.mlb.com/giants/news/very-long-headline-with-many-hyphens-here")
    assert not source_url_allowed("MLB Giants", "https://www.mlb.com/giants/news/video/highlights")


def test_google_wrapper_url_extraction_from_query():
    wrapped = "https://news.google.com/rss/articles/CBMiM2h0dHBzOi8vZXhhbXBsZS5jb20vc3Rvcnk_dXRtX3NvdXJjZT1nb29nbGXSAQA?oc=5&url=https%3A%2F%2Fexample.com%2Fstory"
    assert looks_like_google_wrapper(wrapped)
    assert extract_publisher_url_from_google_wrapper(wrapped) == "https://example.com/story"


def test_source_policy_blocks_fangraphs_category_page():
    allowed, reason = source_policy_allows("Fangraphs Giants", "https://blogs.fangraphs.com/category/giants/", "Giants", "")
    assert not allowed
    assert reason.startswith("blocked_by_source_policy")
