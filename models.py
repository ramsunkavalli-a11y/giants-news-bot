from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Candidate:
    source: str
    url: str
    title: str = ""
    author: str = ""
    summary: str = ""
    categories: List[str] = field(default_factory=list)
    image_url: str = ""
    discovered_via: str = ""
    published_ts: str = ""

    # URL lifecycle fields
    feed_url: str = ""
    resolved_url: str = ""
    publisher_url: str = ""
    canonical_url: str = ""
    post_url: str = ""
    google_url: str = ""

    score: int = 0
    score_components: Dict[str, int] = field(default_factory=dict)
    reject_reasons: List[str] = field(default_factory=list)
    skip_reason: str = ""
    exception: str = ""
    stage: str = "discovered"
    article_meta_confirmed: bool = False
    selected_reasons: List[str] = field(default_factory=list)
    meta_sources_used: List[str] = field(default_factory=list)
    http_status: int = 0
    content_type: str = ""
    is_cardable: bool = False
    validation_domain: str = ""
    source_policy_reason: str = ""

    def add_reject(self, reason: str) -> None:
        self.reject_reasons.append(reason)
        if not self.skip_reason:
            self.skip_reason = reason
