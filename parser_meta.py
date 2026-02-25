from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

from filters import clean_text


@dataclass
class MetaResult:
    title: str = ""
    author: str = ""
    canonical: str = ""
    description: str = ""
    image_url: str = ""


META_NAMES = {
    "author",
    "parsely-author",
    "byl",
    "twitter:description",
    "description",
}


def _as_json(value: str):
    try:
        return json.loads(value)
    except Exception:
        return None


def extract_meta(url: str, html: str) -> MetaResult:
    out = MetaResult()
    if BeautifulSoup is None:
        return out
    soup = BeautifulSoup(html, "lxml")

    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        out.title = clean_text(og_title["content"])
    if not out.title and soup.title:
        out.title = clean_text(soup.title.get_text(" "))

    og_url = soup.find("meta", attrs={"property": "og:url"})
    if og_url and og_url.get("content"):
        out.canonical = urljoin(url, og_url["content"])

    canonical = soup.find("link", attrs={"rel": lambda v: v and "canonical" in str(v).lower()})
    if canonical and canonical.get("href"):
        out.canonical = urljoin(url, canonical["href"])

    og_desc = soup.find("meta", attrs={"property": "og:description"})
    if og_desc and og_desc.get("content"):
        out.description = clean_text(og_desc["content"])

    og_image = soup.find("meta", attrs={"property": lambda v: v and v.startswith("og:image")})
    if og_image and og_image.get("content"):
        out.image_url = urljoin(url, og_image["content"])

    for m in soup.find_all("meta"):
        key = (m.get("name") or "").lower()
        if key not in META_NAMES:
            continue
        val = clean_text(m.get("content") or "")
        if key in {"author", "parsely-author", "byl"} and val and not out.author:
            out.author = val
        if key in {"description", "twitter:description"} and val and not out.description:
            out.description = val

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = (script.string or script.get_text() or "").strip()
        if not raw:
            continue
        data = _as_json(raw)
        if data is None:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            if not out.title and item.get("headline"):
                out.title = clean_text(str(item.get("headline")))
            if not out.description and item.get("description"):
                out.description = clean_text(str(item.get("description")))
            if not out.image_url:
                image = item.get("image")
                if isinstance(image, str):
                    out.image_url = urljoin(url, image)
                elif isinstance(image, list) and image and isinstance(image[0], str):
                    out.image_url = urljoin(url, image[0])
                elif isinstance(image, dict) and image.get("url"):
                    out.image_url = urljoin(url, str(image["url"]))
            if not out.author:
                author = item.get("author")
                if isinstance(author, dict):
                    out.author = clean_text(str(author.get("name", "")))
                elif isinstance(author, list) and author and isinstance(author[0], dict):
                    out.author = clean_text(str(author[0].get("name", "")))
                elif isinstance(author, str):
                    out.author = clean_text(author)

    return out
