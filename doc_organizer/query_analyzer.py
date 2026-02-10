from __future__ import annotations

import json
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


def parse_llm_analysis(message: str, llm_text: str) -> AnalyzedQuery:
    fallback = analyze_user_query(message)
    if not llm_text.strip():
        return fallback

    try:
        data = json.loads(llm_text)
    except json.JSONDecodeError:
        return fallback

    normalized = str(data.get("normalized") or fallback.normalized).strip()
    search_query = str(data.get("search_query") or fallback.search_query).strip()
    raw_keywords = data.get("keywords") or []
    keywords: list[str] = []
    if isinstance(raw_keywords, list):
        for item in raw_keywords:
            if not isinstance(item, str):
                continue
            token = item.strip()
            if token:
                keywords.append(token)

    return AnalyzedQuery(
        original=message,
        normalized=normalized or fallback.normalized,
        keywords=keywords or fallback.keywords,
        search_query=search_query or fallback.search_query,
    )
