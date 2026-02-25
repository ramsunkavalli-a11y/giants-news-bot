from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List, Set
from urllib.parse import urljoin, urlparse

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None
import re
import requests

from config import Settings, SourceConfig
from filters import canonicalize_url, is_story_url
from models import Candidate


def _fetch_text(session: requests.Session, url: str, timeout: int) -> str:
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code >= 400:
            return ""
        return r.text
    except Exception:
        return ""


def discover_sitemaps(session: requests.Session, base_url: str, settings: Settings) -> List[str]:
    root = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}"
    robots = _fetch_text(session, urljoin(root, "/robots.txt"), settings.request_timeout)
    sitemaps = [line.split(":", 1)[1].strip() for line in robots.splitlines() if line.lower().startswith("sitemap:")]
    if not sitemaps:
        sitemaps = [urljoin(root, "/sitemap.xml")]
    return sitemaps[:8]


def urls_from_sitemap(session: requests.Session, sitemap_url: str, settings: Settings) -> List[str]:
    xml = _fetch_text(session, sitemap_url, settings.request_timeout)
    if not xml:
        return []
    out: List[str] = []
    try:
        root = ET.fromstring(xml.encode("utf-8"))
    except Exception:
        return out
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    for loc in root.findall(".//sm:url/sm:loc", ns):
        if loc.text:
            out.append(loc.text.strip())
    return out


def discover_from_listing(session: requests.Session, listing_url: str, settings: Settings) -> List[str]:
    html = _fetch_text(session, listing_url, settings.request_timeout)
    if not html:
        return []
    found = set()
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a"):
            href = a.get("href")
            if not href:
                continue
            abs_url = canonicalize_url(urljoin(listing_url, href), settings)
            if abs_url.startswith("http"):
                found.add(abs_url)
    else:
        for href in re.findall(r'<a[^>]+href=["\']([^"\']+)["\']', html, flags=re.I):
            abs_url = canonicalize_url(urljoin(listing_url, href), settings)
            if abs_url.startswith("http"):
                found.add(abs_url)
    return list(found)


def discover_non_rss(settings: Settings, session: requests.Session, sources: List[SourceConfig]) -> List[Candidate]:
    out: List[Candidate] = []
    for source in sources:
        seen: Set[str] = set()
        filtered: List[str] = []

        for sm in discover_sitemaps(session, source.listing_url, settings)[:3]:
            for u in urls_from_sitemap(session, sm, settings):
                cu = canonicalize_url(u, settings)
                if not cu or cu in seen:
                    continue
                seen.add(cu)
                if is_story_url(cu):
                    filtered.append(cu)
                if len(filtered) >= settings.max_non_rss_urls_per_source:
                    break
            if len(filtered) >= settings.max_non_rss_urls_per_source:
                break

        if not filtered:
            for u in discover_from_listing(session, source.listing_url, settings):
                cu = canonicalize_url(u, settings)
                if not cu or cu in seen:
                    continue
                seen.add(cu)
                if is_story_url(cu):
                    filtered.append(cu)
                if len(filtered) >= settings.max_non_rss_urls_per_source:
                    break

        for url in filtered:
            out.append(Candidate(source=source.name, url=url, discovered_via="nonrss"))
    return out
