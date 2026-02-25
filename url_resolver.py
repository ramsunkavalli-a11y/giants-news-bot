from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import parse_qsl, urljoin, urlparse

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

from filters import (
    canonicalize_url,
    extract_external_url_from_text,
    extract_publisher_url_from_google_wrapper,
    is_story_url,
    looks_like_google_wrapper,
)
from parser_meta import extract_meta


@dataclass
class ResolutionResult:
    resolved_url: str = ""
    canonical_url: str = ""
    post_url: str = ""
    http_status: int = 0
    content_type: str = ""
    validation_domain: str = ""
    resolver_path: str = ""
    exception: str = ""
    is_cardable: bool = False
    article_meta_confirmed: bool = False
    failure_reason: str = ""
    meta_sources_used: list[str] = field(default_factory=list)


def _extract_meta_refresh_target(base_url: str, html: str) -> str:
    if BeautifulSoup is None:
        return ""
    soup = BeautifulSoup(html, "lxml")
    tag = soup.find("meta", attrs={"http-equiv": lambda v: v and str(v).lower() == "refresh"})
    if not tag:
        return ""
    content = tag.get("content", "")
    lower = content.lower()
    if "url=" not in lower:
        return ""
    url_part = content[lower.index("url=") + 4 :].strip()
    return urljoin(base_url, url_part)


def _extract_url_from_google_html(base_url: str, html: str) -> str:
    meta = extract_meta(base_url, html)
    if BeautifulSoup is None:
        return meta.canonical or ""
    if meta.canonical:
        return urljoin(base_url, meta.canonical)
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http") and "news.google.com" not in href:
            return href
    refresh = _extract_meta_refresh_target(base_url, html)
    if refresh:
        return refresh
    return ""


def resolve_article_url(candidate, session, settings, verbose: bool = False) -> ResolutionResult:
    result = ResolutionResult()
    feed_url = canonicalize_url(candidate.feed_url or candidate.url, settings)
    if not feed_url:
        result.failure_reason = "bad_feed_url"
        return result

    # direct non-google URL fast path
    if not looks_like_google_wrapper(feed_url):
        result.resolved_url = feed_url
        result.canonical_url = feed_url
        result.post_url = feed_url
        result.validation_domain = urlparse(feed_url).netloc.lower()
        result.resolver_path = "direct_feed_url"
        result.is_cardable = feed_url.startswith("https://") and "news.google.com" not in feed_url
        return result

    # wrapper path
    hinted = extract_external_url_from_text(candidate.summary)
    if hinted:
        hinted = canonicalize_url(hinted, settings)
        result.resolved_url = hinted
        result.resolver_path = "summary_url_hint"

    if not result.resolved_url:
        hinted2 = extract_publisher_url_from_google_wrapper(feed_url)
        if hinted2:
            result.resolved_url = canonicalize_url(hinted2, settings)
            result.resolver_path = "wrapper_param_decode"

    html = ""
    try:
        r = session.get(feed_url, timeout=settings.request_timeout, allow_redirects=True)
        result.http_status = r.status_code
        result.content_type = r.headers.get("Content-Type", "")
        final = canonicalize_url(r.url or "", settings)
        if final and "news.google.com" not in final:
            result.resolved_url = final
            result.resolver_path = result.resolver_path or "redirect_final"
        if "text/html" in result.content_type:
            html = r.text[:800_000]
    except Exception as exc:
        result.exception = repr(exc)

    if html:
        meta = extract_meta(result.resolved_url or feed_url, html)
        result.meta_sources_used = meta.meta_sources_used
        extracted = _extract_url_from_google_html(result.resolved_url or feed_url, html)
        if extracted:
            result.canonical_url = canonicalize_url(extracted, settings)
            if "news.google.com" not in result.canonical_url:
                result.resolver_path = result.resolver_path or "google_html_extract"
        if meta.canonical:
            can = canonicalize_url(urljoin(result.resolved_url or feed_url, meta.canonical), settings)
            if can and "news.google.com" not in can:
                result.canonical_url = can
                result.resolver_path = result.resolver_path or "meta_canonical"
        result.article_meta_confirmed = meta.article_meta_confirmed

    if not result.canonical_url:
        result.canonical_url = result.resolved_url

    # final decode fallback if still google
    if (not result.canonical_url) or ("news.google.com" in result.canonical_url):
        hinted3 = extract_publisher_url_from_google_wrapper(feed_url)
        if hinted3:
            hinted3 = canonicalize_url(hinted3, settings)
            if hinted3 and "news.google.com" not in hinted3:
                result.canonical_url = hinted3
                result.resolver_path = result.resolver_path or "fallback_decode"

    result.post_url = result.canonical_url or result.resolved_url
    result.validation_domain = urlparse(result.post_url or feed_url).netloc.lower()
    result.is_cardable = bool(
        result.post_url
        and result.post_url.startswith("https://")
        and "news.google.com" not in result.post_url
        and is_story_url(result.post_url)
    )

    if not result.post_url:
        result.failure_reason = "unresolved_google_news_redirect"
    elif "news.google.com" in result.post_url:
        result.failure_reason = "unresolved_google_news_redirect"
    elif not is_story_url(result.post_url):
        result.failure_reason = "non_article_page"
    elif result.exception and not result.http_status:
        result.failure_reason = "network_error"

    if verbose:
        print(
            f"resolver debug feed_url={feed_url} path={result.resolver_path} "
            f"post_url={result.post_url} failure={result.failure_reason}"
        )

    return result
