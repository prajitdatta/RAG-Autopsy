"""
failures/f02_retrieval/test_embedding_model_mismatch.py

FAILURE: Index built with one embedding model, queries run with another.

ROOT CAUSE:
    Every embedding model maps text to its own vector space. Vectors from
    model A are not comparable to vectors from model B. If you:
    - Build the index with text-embedding-ada-002
    - Then upgrade to text-embedding-3-small
    - And query WITHOUT rebuilding the index

    ...every query will return garbage results. The cosine similarities are
    computed in mismatched spaces and are meaningless.

    This happens more often than you'd think:
    - Model deprecated, you swap to the new one without rebuilding
    - Index built on Friday, model upgraded over the weekend
    - Two microservices using different embedding model versions

IMPACT:
    Silent failure — no error is thrown, retrieval just returns wrong results.
    Often undetected until a user complains weeks later.

RUN:
    pytest failures/f02_retrieval/test_embedding_model_mismatch.py -v
"""

import pytest
import math
import re
from dataclasses import dataclass
from src.retriever import CosineRetriever, RetrievalResult
from data.sample_docs import get_all_docs
from data.sample_queries import QUERIES


# ── Simulate two different embedding models ───────────────────────────────────

class ModelV1Embedder:
    """
    Simulates embedding model v1 (e.g. text-embedding-ada-002).
    Uses word-count TF — straightforward frequency-based vectors.
    """
    name = "model-v1"

    def embed(self, text: str) -> dict[str, float]:
        tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
        vec: dict[str, float] = {}
        for t in tokens:
            vec[t] = vec.get(t, 0) + 1
        total = sum(vec.values()) or 1
        vec = {k: v / total for k, v in vec.items()}
        return self._normalize(vec)

    @staticmethod
    def _normalize(vec):
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1e-9
        return {k: v / norm for k, v in vec.items()}


class ModelV2Embedder:
    """
    Simulates embedding model v2 (e.g. text-embedding-3-small).
    Uses DIFFERENT tokenization and weighting — chars instead of words,
    with a different normalization scheme. Vectors live in a DIFFERENT space.
    """
    name = "model-v2"

    def embed(self, text: str) -> dict[str, float]:
        # V2 uses bigrams and gives extra weight to less common terms
        tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
        # Bigrams (different representation)
        bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        all_tokens = tokens + bigrams
        vec: dict[str, float] = {}
        for t in all_tokens:
            vec[t] = vec.get(t, 0) + 1
        # Different normalization — log-scale
        vec = {k: math.log(v + 1) for k, v in vec.items()}
        return self._normalize(vec)

    @staticmethod
    def _normalize(vec):
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1e-9
        return {k: v / norm for k, v in vec.items()}


def cosine(a: dict, b: dict) -> float:
    return sum(a.get(k, 0) * v for k, v in b.items())


class MismatchedRetriever:
    """
    Retriever that indexes with model_v1 but queries with model_v2.
    This is the bug.
    """

    def __init__(self):
        self._docs = []
        self._vectors = []
        self._index_embedder = ModelV1Embedder()  # Used at index time
        self._query_embedder = ModelV2Embedder()  # Used at query time — MISMATCH

    def index(self, documents: list[dict]):
        self._docs = documents
        self._vectors = [self._index_embedder.embed(d["text"]) for d in documents]

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        query_vec = self._query_embedder.embed(query)  # Wrong model!
        scores = [
            (cosine(query_vec, doc_vec), i)
            for i, doc_vec in enumerate(self._vectors)
        ]
        scores.sort(reverse=True)
        return [
            RetrievalResult(
                chunk_id=self._docs[i]["id"],
                text=self._docs[i]["text"],
                score=round(s, 4),
                rank=rank + 1,
                retriever="mismatched",
            )
            for rank, (s, i) in enumerate(scores[:top_k])
        ]


class CorrectRetriever:
    """Same model for both indexing and querying — correct."""

    def __init__(self):
        self._docs = []
        self._vectors = []
        self._embedder = ModelV1Embedder()

    def index(self, documents: list[dict]):
        self._docs = documents
        self._vectors = [self._embedder.embed(d["text"]) for d in documents]

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        query_vec = self._embedder.embed(query)
        scores = [(cosine(query_vec, doc_vec), i) for i, doc_vec in enumerate(self._vectors)]
        scores.sort(reverse=True)
        return [
            RetrievalResult(
                chunk_id=self._docs[i]["id"],
                text=self._docs[i]["text"],
                score=round(s, 4),
                rank=rank + 1,
                retriever="matched",
            )
            for rank, (s, i) in enumerate(scores[:top_k])
        ]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEmbeddingModelMismatch:

    def test_mismatched_models_degrade_retrieval(self):
        """
        FAILURE: Building the index with model-v1 and querying with model-v2
        produces retrieval results no better than random ordering.
        """
        docs = get_all_docs()
        doc_list = [{"id": d["id"], "text": d["content"]} for d in docs]

        correct    = CorrectRetriever()
        mismatched = MismatchedRetriever()
        correct.index(doc_list)
        mismatched.index(doc_list)

        correct_hits    = 0
        mismatched_hits = 0
        total = 0

        for q in QUERIES[:6]:
            correct_results    = correct.retrieve(q["query"],    top_k=3)
            mismatched_results = mismatched.retrieve(q["query"], top_k=3)

            correct_top3    = [r.chunk_id for r in correct_results]
            mismatched_top3 = [r.chunk_id for r in mismatched_results]

            # Hit = source document appears in top-3
            if any(q["source_doc"] in cid for cid in correct_top3):
                correct_hits += 1
            if any(q["source_doc"] in cid for cid in mismatched_top3):
                mismatched_hits += 1
            total += 1

        correct_recall    = correct_hits / total
        mismatched_recall = mismatched_hits / total

        print(f"\n[FAILURE DEMO] Retrieval recall (source doc in top-3):")
        print(f"  Matched models (correct):    {correct_hits}/{total} ({correct_recall:.0%})")
        print(f"  Mismatched models (broken):  {mismatched_hits}/{total} ({mismatched_recall:.0%})")

        assert correct_recall >= mismatched_recall, (
            "Correctly matched models should produce equal or better recall."
        )

    def test_cosine_scores_are_meaningless_across_models(self):
        """
        FAILURE: Cosine similarity scores between v1 index vectors and v2 query
        vectors are near-zero or negative — they live in different spaces.
        """
        text = "Refunds are processed within 5-7 business days."
        query = "How long does a refund take?"

        v1 = ModelV1Embedder()
        v2 = ModelV2Embedder()

        v1_doc_vec   = v1.embed(text)
        v1_query_vec = v1.embed(query)
        v2_query_vec = v2.embed(query)

        sim_matched   = cosine(v1_query_vec, v1_doc_vec)
        sim_mismatched = cosine(v2_query_vec, v1_doc_vec)

        print(f"\n[FAILURE DEMO] Cosine similarity:")
        print(f"  v1 query vs v1 doc (correct):    {sim_matched:.4f}")
        print(f"  v2 query vs v1 doc (mismatched): {sim_mismatched:.4f}")

        assert sim_matched > sim_mismatched, (
            "Matched model similarity should be higher than mismatched."
        )

    def test_no_error_raised_on_mismatch(self):
        """
        FAILURE: The mismatch produces NO error. It silently returns wrong results.
        This is why it's so dangerous — you won't catch it without an eval harness.
        """
        docs = get_all_docs()
        mismatched = MismatchedRetriever()
        mismatched.index([{"id": d["id"], "text": d["content"]} for d in docs])

        # This should raise an error but it doesn't — silent failure
        try:
            results = mismatched.retrieve("What is the return policy?", top_k=3)
            print(f"\n[FAILURE DEMO] No error raised despite model mismatch.")
            print(f"  Returned {len(results)} results silently.")
            print(f"  Top result: [{results[0].chunk_id}] score={results[0].score:.4f}")
            silent_failure = True
        except Exception as e:
            silent_failure = False
            print(f"  Error raised (unexpected): {e}")

        assert silent_failure, "Model mismatch should silently return wrong results (no error)"

    def test_fix_rebuild_index_after_model_change(self):
        """
        FIX: Always rebuild the entire index when the embedding model changes.
        Document this requirement explicitly in your deployment runbook.
        """
        docs = get_all_docs()
        doc_list = [{"id": d["id"], "text": d["content"]} for d in docs]

        # Simulate: v1 used before, v2 used now — but we REBUILD
        v2_correct = CorrectRetriever()
        v2_correct._embedder = ModelV2Embedder()   # Upgrade to v2
        v2_correct.index(doc_list)                 # Rebuild with v2

        # Query also uses v2 — consistent
        results = v2_correct.retrieve(QUERIES[0]["query"], top_k=3)
        assert len(results) > 0, "Rebuilt index should return results"

        print(f"\n[FIX VERIFICATION] Index rebuilt with v2 — retrieval works:")
        for r in results:
            print(f"  Rank {r.rank}: [{r.chunk_id}] score={r.score:.4f}")
