"""
fixes/fix_retrieval.py

Production-ready retrieval fixes for F02 failures.
"""

import re
from src.retriever import HybridRetriever, RetrievalResult


class ProductionRetriever:
    """
    Retriever with built-in fixes for common F02 failure modes.

    Fixes:
        F02a: Uses hybrid (BM25 + cosine) — partial semantic mismatch fix
        F02b: Enforces relevance threshold — no-fallback fix
        F02c: Deduplicates results — duplicate chunk fix
        F02d: Reranks with cross-encoder simulation
    """

    def __init__(
        self,
        top_k: int = 5,
        relevance_threshold: float = 0.01,
        dedup_threshold: float = 0.7,
    ):
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        self.dedup_threshold = dedup_threshold
        self._retriever = HybridRetriever()

    def index(self, documents: list[dict]):
        self._retriever.index(documents)

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """
        Retrieve with relevance gating, deduplication, and reranking.
        Returns empty list if no relevant results found (triggers fallback).
        """
        candidates = self._retriever.retrieve(query, top_k=self.top_k * 2)

        # Gate on relevance threshold
        relevant = [r for r in candidates if r.score >= self.relevance_threshold]

        if not relevant:
            return []  # Signal to caller: no relevant content found

        # Deduplicate
        deduplicated = self._deduplicate(relevant)

        # Simple rerank: promote chunks with more query term matches
        reranked = self._rerank(query, deduplicated)

        return reranked[:self.top_k]

    def _deduplicate(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        kept = []
        for r in results:
            is_dup = any(self._jaccard(r.text, k.text) >= self.dedup_threshold for k in kept)
            if not is_dup:
                kept.append(r)
        return kept

    def _rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        query_terms = set(re.findall(r"\b[a-z]{3,}\b", query.lower()))
        scored = []
        for r in results:
            text_tokens = re.findall(r"\b[a-z]{3,}\b", r.text.lower())
            tf = sum(text_tokens.count(t) for t in query_terms)
            scored.append((tf, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        for new_rank, (_, r) in enumerate(scored):
            r.rank = new_rank + 1
        return [r for _, r in scored]

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        ta = set(re.findall(r"\b[a-z]{3,}\b", a.lower()))
        tb = set(re.findall(r"\b[a-z]{3,}\b", b.lower()))
        union = ta | tb
        return len(ta & tb) / len(union) if union else 0.0
