from models import Candidate
from config import Settings
from url_resolver import resolve_article_url


class FakeResponse:
    def __init__(self, url, status=200, content_type="text/html", text=""):
        self.url = url
        self.status_code = status
        self.headers = {"Content-Type": content_type}
        self.text = text


class FakeSession:
    def __init__(self, response: FakeResponse):
        self._response = response

    def get(self, *args, **kwargs):
        return self._response


def test_google_wrapper_resolves_from_summary_hint():
    settings = Settings()
    c = Candidate(
        source="Mercury News Giants",
        url="https://news.google.com/rss/articles/abc",
        feed_url="https://news.google.com/rss/articles/abc",
        summary="Read: https://www.mercurynews.com/2026/02/24/sf-giants-rotation-depth/",
    )
    session = FakeSession(FakeResponse("https://news.google.com/rss/articles/abc", text="<html></html>"))
    res = resolve_article_url(c, session, settings)
    assert "news.google.com" not in (res.post_url or "")
    assert res.resolver_path in {"summary_url_hint", "wrapper_param_decode", "fallback_decode"}


def test_google_wrapper_unresolved_failure():
    settings = Settings()
    c = Candidate(
        source="SF Chronicle Giants",
        url="https://news.google.com/rss/articles/abc",
        feed_url="https://news.google.com/rss/articles/abc",
    )
    session = FakeSession(FakeResponse("https://news.google.com/rss/articles/abc", text="<html><body>No link</body></html>"))
    res = resolve_article_url(c, session, settings)
    assert res.failure_reason in {"unresolved_google_news_redirect", "non_article_page"}


def test_ap_hub_blocked_by_source_policy():
    from filters import source_policy_allows

    allowed, reason = source_policy_allows("AP Giants", "https://apnews.com/hub/san-francisco-giants", "AP Sports", "")
    assert not allowed
    assert reason == "blocked_ap_non_article"
