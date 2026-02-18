"""
failures/f02_retrieval/test_missing_reranking.py

FAILURE: Retrieval returns chunks ordered by embedding similarity, not relevance.
         Without a reranker, the most-similar chunk ≠ most-relevant chunk.

ROOT CAUSE:
    Embedding similarity measures vector proximity. Reranking measures
    answer-quality. A chunk can be highly similar to a query (shares vocabulary)
    without containing the actual answer. Without reranking:

    1. The "best" chunk by cosine similarity may be a definition section
       that *mentions* the topic but doesn't *answer* the question
    2. The actually-correct chunk may be ranked 3rd or 4th
    3. The LLM receives the wrong chunk first and often anchors on it

IMPACT:
    Precision@1 (is the top chunk correct?) is far lower without reranking.
    For k=1 pipelines this is catastrophic.

RUN:
    pytest failures/f02_retrieval/test_missing_reranking.py -v
"""

import pytest
from src.retriever import HybridRetriever, BM25Retriever
from data.sample_docs import get_all_docs
from data.sample_queries import QUERIES


class MockCrossEncoderReranker:
    """
    Simulated cross-encoder reranker.
    Scores (query, chunk) pairs by counting query-term presence
    and proximity (how close together the query terms appear in the chunk).

    In production: replace with cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    def rerank(
        self,
        query: str,
        candidates: list,
        top_k: int | None = None,
    ) -> list:
        """
        Rerank retrieved candidates by query-document relevance.

        Args:
            query:      The original query
            candidates: List of RetrievalResult objects
            top_k:      Return only top_k after reranking

        Returns:
            Reranked list of RetrievalResult (new .rank values)
        """
        import re

        query_terms = set(re.findall(r"\b[a-z]{3,}\b", query.lower()))

        scored = []
        for candidate in candidates:
            text_lower = candidate.text.lower()
            tokens = re.findall(r"\b[a-z]{3,}\b", text_lower)

            # Score 1: term frequency in chunk
            tf_score = sum(tokens.count(t) for t in query_terms)

            # Score 2: proximity bonus — are all terms within 50 tokens of each other?
            positions = []
            for t in query_terms:
                if t in tokens:
                    positions.append(tokens.index(t))
            proximity_score = 0
            if len(positions) >= 2:
                span = max(positions) - min(positions)
                proximity_score = max(0, 50 - span)

            # Score 3: exact phrase bonus
            phrase = " ".join(sorted(query_terms)[:3])
            phrase_score = 10 if phrase in text_lower else 0

            total_score = tf_score + proximity_score * 0.5 + phrase_score
            scored.append((total_score, candidate))

        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = []
        for new_rank, (score, result) in enumerate(scored):
            result.rank = new_rank + 1
            result.score = round(score, 4)
            result.retriever = f"{result.retriever}+reranked"
            reranked.append(result)

        return reranked[:top_k] if top_k else reranked


class TestMissingReranking:

    def test_top1_without_reranking_often_misses_answer(self):
        """
        FAILURE: The top-ranked chunk by BM25/cosine score is not always the one
        containing the actual answer. Reranking fixes rank order.
        """
        docs = get_all_docs()
        retriever = HybridRetriever()
        retriever.index([{"id": d["id"], "text": d["content"]} for d in docs])
        reranker = MockCrossEncoderReranker()

        rank_improvements = []

        for query_data in QUERIES[:6]:
            query      = query_data["query"]
            key_fact   = query_data["key_facts"][0]
            source_doc = query_data["source_doc"]

            candidates_no_rerank  = retriever.retrieve(query, top_k=8)
            candidates_reranked   = reranker.rerank(query, retriever.retrieve(query, top_k=8), top_k=5)

            # Find rank of the chunk containing the key fact
            def find_fact_rank(results, fact):
                for r in results:
                    if fact.lower() in r.text.lower():
                        return r.rank
                return len(results) + 1

            rank_before = find_fact_rank(candidates_no_rerank, key_fact)
            rank_after  = find_fact_rank(candidates_reranked, key_fact)
            rank_improvements.append(rank_before - rank_after)

            print(f"\n  Query: '{query[:55]}'")
            print(f"    Key fact: '{key_fact}'")
            print(f"    Rank without reranking: {rank_before}")
            print(f"    Rank with reranking:    {rank_after}")
            print(f"    Improvement: {rank_before - rank_after:+d}")

        improved = sum(1 for i in rank_improvements if i > 0)
        stayed   = sum(1 for i in rank_improvements if i == 0)
        worse    = sum(1 for i in rank_improvements if i < 0)

        print(f"\n[FAILURE DEMO] Rank changes after reranking:")
        print(f"  Improved: {improved}, Unchanged: {stayed}, Worse: {worse}")

    def test_precision_at_1_improves_with_reranking(self):
        """
        FAILURE: Precision@1 (correct answer in top-1 chunk) is lower without reranking.
        Demonstrates the concrete quality benefit of reranking.
        """
        docs = get_all_docs()
        retriever = HybridRetriever()
        retriever.index([{"id": d["id"], "text": d["content"]} for d in docs])
        reranker = MockCrossEncoderReranker()

        p1_no_rerank = 0
        p1_reranked  = 0
        total = 0

        for query_data in QUERIES:
            query    = query_data["query"]
            key_fact = query_data["key_facts"][0]
            total   += 1

            no_rerank = retriever.retrieve(query, top_k=5)
            reranked  = reranker.rerank(query, retriever.retrieve(query, top_k=5))

            if no_rerank and key_fact.lower() in no_rerank[0].text.lower():
                p1_no_rerank += 1
            if reranked and key_fact.lower() in reranked[0].text.lower():
                p1_reranked += 1

        p1_no_rerank_rate = p1_no_rerank / total
        p1_reranked_rate  = p1_reranked / total

        print(f"\n[FAILURE DEMO] Precision@1 (key fact in top-1 chunk):")
        print(f"  Without reranking: {p1_no_rerank}/{total} ({p1_no_rerank_rate:.0%})")
        print(f"  With reranking:    {p1_reranked}/{total} ({p1_reranked_rate:.0%})")

        assert p1_reranked_rate >= p1_no_rerank_rate, (
            "Reranking should achieve equal or better Precision@1."
        )

    def test_reranker_promotes_direct_answer_over_related_topic(self):
        """
        FAILURE: Without reranking, a chunk that merely mentions 'refund' (topic)
        can outrank a chunk that *answers* 'how long does a refund take' (answer).
        """
        docs = get_all_docs()
        retriever = BM25Retriever()
        retriever.index([{"id": d["id"], "text": d["content"]} for d in docs])
        reranker = MockCrossEncoderReranker()

        query = "How long does a refund take to process?"
        answer_fragment = "5-7 business days"

        no_rerank = retriever.retrieve(query, top_k=10)
        reranked  = reranker.rerank(query, retriever.retrieve(query, top_k=10))

        # Find where the answer_fragment appears
        def answer_rank(results):
            for r in results:
                if answer_fragment.lower() in r.text.lower():
                    return r.rank
            return 999

        rank_before = answer_rank(no_rerank)
        rank_after  = answer_rank(reranked)

        print(f"\n[FAILURE DEMO] Rank of chunk containing '{answer_fragment}':")
        print(f"  Without reranking: {rank_before}")
        print(f"  With reranking:    {rank_after}")

        print(f"\n  Without reranking — top 3 chunks:")
        for r in no_rerank[:3]:
            has_answer = "✅" if answer_fragment in r.text else "❌"
            print(f"    {has_answer} Rank {r.rank}: '{r.text[:70]}...'")

        print(f"\n  With reranking — top 3 chunks:")
        for r in reranked[:3]:
            has_answer = "✅" if answer_fragment in r.text else "❌"
            print(f"    {has_answer} Rank {r.rank}: '{r.text[:70]}...'")
