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
    canonical_url: str = ""
    score: int = 0
    score_components: Dict[str, int] = field(default_factory=dict)
    reject_reasons: List[str] = field(default_factory=list)
    article_meta_confirmed: bool = False
    selected_reasons: List[str] = field(default_factory=list)

    def add_reject(self, reason: str) -> None:
        self.reject_reasons.append(reason)
