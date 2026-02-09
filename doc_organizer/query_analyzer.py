from __future__ import annotations

import re
from dataclasses import dataclass

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "what",
    "where",
    "when",
    "how",
    "왜",
    "어떻게",
    "무엇",
    "뭐",
    "이거",
    "저거",
    "관련",
    "대해",
    "에서",
    "으로",
    "이다",
}


@dataclass(slots=True)
class AnalyzedQuery:
    original: str
    normalized: str
    keywords: list[str]
    search_query: str


def analyze_user_query(message: str) -> AnalyzedQuery:
    normalized = re.sub(r"\s+", " ", message).strip()
    tokens = re.findall(r"[A-Za-z0-9가-힣]{2,}", normalized.lower())
    keywords: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in STOPWORDS:
            continue
        if token in seen:
            continue
        keywords.append(token)
        seen.add(token)

    if keywords:
        search_query = f"{normalized} {' '.join(keywords[:6])}"
    else:
        search_query = normalized

    return AnalyzedQuery(
        original=message,
        normalized=normalized,
        keywords=keywords,
        search_query=search_query,
    )

