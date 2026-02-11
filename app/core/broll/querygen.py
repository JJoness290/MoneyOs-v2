from __future__ import annotations

import re
from collections import Counter

STOPWORDS = {
    "the",
    "and",
    "to",
    "of",
    "a",
    "in",
    "is",
    "it",
    "that",
    "for",
    "on",
    "with",
    "as",
    "are",
    "was",
    "be",
    "by",
    "this",
    "an",
    "or",
    "from",
    "at",
    "we",
    "you",
    "your",
    "our",
    "they",
    "their",
    "not",
}

BROAD_TOPICS = {
    "money": "money",
    "finance": "finance",
    "invest": "investment",
    "budget": "budget",
    "business": "business meeting",
    "office": "office work",
    "city": "city night",
    "tech": "technology",
    "motivation": "motivation",
}

DOMAIN_ANCHORS = {
    "finance_legal": [
        "finance audit legal",
        "bank transfer",
        "audit documents",
        "legal contract",
        "city council meeting",
        "escrow paperwork",
        "bank transfer paperwork",
    ],
}


def detect_domain(script_text: str) -> str:
    terms = _extract_terms(script_text)
    domain_terms = {"finance", "audit", "escrow", "contract", "legal", "court", "bank"}
    if any(term in domain_terms for term in terms):
        return "finance_legal"
    return "general"

def _extract_terms(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def _detect_topic(terms: list[str]) -> str:
    for term in terms:
        if term in BROAD_TOPICS:
            return BROAD_TOPICS[term]
    return "business meeting"


def build_queries(segment_text: str, domain: str, max_queries: int = 6) -> list[str]:
    terms = _extract_terms(segment_text)
    counts = Counter(terms)
    common = [term for term, _ in counts.most_common(8)]
    queries = []
    if common:
        if len(common) >= 2:
            queries.append(" ".join(common[:2]))
        if len(common) >= 4:
            queries.append(" ".join(common[:3]))
        for term in common[:4]:
            if term not in queries:
                queries.append(term)
    queries.append(_detect_topic(terms))
    if domain in DOMAIN_ANCHORS:
        queries.extend(DOMAIN_ANCHORS[domain])
    if not queries:
        queries.append(_detect_topic(terms))
    deduped = []
    for query in queries:
        query = query.strip()
        if query and query not in deduped:
            deduped.append(query)
    return deduped[:max_queries]
