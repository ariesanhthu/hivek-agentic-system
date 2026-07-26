"""Small pure-Python TF-IDF retriever for approved reply candidates."""

from __future__ import annotations

import math
from collections import Counter
from typing import Literal

from pydantic import BaseModel

from hivek_agent.reply.normalization import tokens


class ReplyCandidate(BaseModel):
    candidate_id: str
    match_text: str
    reply_text: str
    source_type: Literal["post_reply_suggestion", "confirmed_fact", "approved_reply"]
    source_id: str


class RankedCandidate(BaseModel):
    candidate: ReplyCandidate
    score: float


def rank_candidates(
    query: str, candidates: list[ReplyCandidate], *, limit: int = 5
) -> list[RankedCandidate]:
    if not candidates:
        return []
    documents = [tokens(candidate.match_text) for candidate in candidates]
    query_terms = tokens(query)
    if not query_terms:
        return []

    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(document))
    count = len(documents)
    idf = {
        term: math.log((count + 1) / (frequency + 1)) + 1
        for term, frequency in document_frequency.items()
    }
    # Query-only words get the strongest smoothed IDF; they still cannot contribute to
    # a document dot product, but including them correctly lowers cosine similarity.
    query_vector = _vector(query_terms, idf, fallback_idf=math.log(count + 1) + 1)
    ranked: list[RankedCandidate] = []
    for candidate, document in zip(candidates, documents, strict=True):
        score = _cosine(query_vector, _vector(document, idf, fallback_idf=1.0))
        ranked.append(RankedCandidate(candidate=candidate, score=round(score, 6)))
    ranked.sort(key=lambda item: item.score, reverse=True)
    return [item for item in ranked[:limit] if item.score > 0]


def _vector(terms: list[str], idf: dict[str, float], *, fallback_idf: float) -> dict[str, float]:
    counts = Counter(terms)
    total = max(1, len(terms))
    return {
        term: (frequency / total) * idf.get(term, fallback_idf)
        for term, frequency in counts.items()
    }


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    dot = sum(weight * right.get(term, 0.0) for term, weight in left.items())
    left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
    right_norm = math.sqrt(sum(weight * weight for weight in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)
