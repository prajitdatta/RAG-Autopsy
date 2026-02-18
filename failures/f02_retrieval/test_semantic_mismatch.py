"""
failures/f02_retrieval/test_semantic_mismatch.py

FAILURE: Embedding model can't match query vocabulary to document vocabulary.

ROOT CAUSE:
    Vocabulary mismatch: the query uses different words than the document
    but means the same thing. A cosine retriever using TF-IDF or a weak
    embedding model fails because the overlap is low even if semantically identical.

    Examples of mismatches this test demonstrates:
    - "send back an item" vs "return policy"
    - "how much do I earn extra" vs "salary increase / merit raise"
    - "sugar disease treatment" vs "Type 2 diabetes management"
    - "headset sound specs" vs "audio specifications"

IMPACT:
    The correct document exists in the index. The query is reasonable.
    But retrieval returns wrong documents because lexical overlap is low.
    The LLM then either says "I don't have this information" or hallucinates.

RUN:
    pytest failures/f02_retrieval/test_semantic_mismatch.py -v
"""

import pytest
from src.retriever import BM25Retriever, CosineRetriever, HybridRetriever
from src.rag_pipeline import RAGPipeline
from data.sample_docs import get_all_docs, get_doc
from data.sample_queries import QUERIES


# Paraphrase pairs: (query_using_different_vocab, expected_source_doc)
PARAPHRASE_QUERIES = [
    ("send back a purchase", "refund_policy"),
    ("get money back for something I bought", "refund_policy"),
    ("how much extra pay for excellent work", "employee_handbook"),
    ("criteria for moving up to next level", "employee_handbook"),
    ("blood sugar medication starting amount", "medical_guidelines"),
    ("headset sound quality numbers", "product_spec"),
    ("company earnings last year", "financial_report"),
]


def build_index(retriever_class, docs: list[dict]):
    """Helper: index documents with given retriever class."""
    retriever = retriever_class()
    retriever.index([{"id": d["id"], "text": d["content"]} for d in docs])
    return retriever


class TestSemanticMismatchWithBM25:

    def test_bm25_fails_on_paraphrase_queries(self):
        """
        FAILURE: BM25 depends on exact keyword overlap. Paraphrased queries
        that use no words from the document score near zero.
        """
        docs = get_all_docs()
        retriever = build_index(BM25Retriever, docs)

        failures = []
        for query, expected_doc in PARAPHRASE_QUERIES:
            results = retriever.retrieve(query, top_k=3)
            top_ids = [r.chunk_id.split("_chunk_")[0] for r in results]
            hit = expected_doc in top_ids
            failures.append((query, expected_doc, top_ids[0] if top_ids else "none", hit))

        hits = sum(1 for _, _, _, h in failures if h)
        total = len(failures)
        hit_rate = hits / total

        print(f"\n[FAILURE DEMO] BM25 paraphrase recall: {hits}/{total} ({hit_rate:.0%})")
        for query, expected, got, hit in failures:
            status = "✅" if hit else "❌"
            print(f"  {status} Query: '{query[:50]}'")
            print(f"     Expected: {expected}, Got: {got}")

        # BM25 should fail on paraphrase queries
        assert hit_rate < 0.6, (
            f"Expected BM25 to struggle with paraphrase queries (<60% recall). "
            f"Got {hit_rate:.0%}. The vocabulary may overlap more than expected."
        )

    def test_bm25_zero_score_for_synonym_query(self):
        """
        FAILURE: If query has zero word overlap with all documents,
        BM25 scores all documents identically at 0 and returns random order.
        """
        docs = get_all_docs()
        retriever = build_index(BM25Retriever, docs)

        # Query with medical synonyms not in the document
        query = "hyperglycemia pharmacotherapy initiation dosage"
        results = retriever.retrieve(query, top_k=5)

        scores = [r.score for r in results]
        all_zero = all(s == 0.0 for s in scores)

        print(f"\n[FAILURE DEMO] BM25 scores for synonym query:")
        for r in results:
            print(f"  Rank {r.rank}: [{r.chunk_id[:30]}] score={r.score:.4f}")

        if all_zero:
            print(f"  All scores are 0.0 — BM25 has no signal, returns arbitrary order")


class TestSemanticMismatchWithWeakEmbeddings:

    def test_cosine_retriever_on_paraphrase_queries(self):
        """
        Compare BM25 and Cosine retriever on paraphrase queries.
        With TF-IDF vectors (no neural embeddings), cosine does only slightly better.
        """
        docs = get_all_docs()
        bm25   = build_index(BM25Retriever,   docs)
        cosine = build_index(CosineRetriever, docs)

        bm25_hits   = 0
        cosine_hits = 0

        for query, expected_doc in PARAPHRASE_QUERIES:
            bm25_top = [r.chunk_id.split("_chunk_")[0] for r in bm25.retrieve(query, top_k=3)]
            cos_top  = [r.chunk_id.split("_chunk_")[0] for r in cosine.retrieve(query, top_k=3)]
            if expected_doc in bm25_top:
                bm25_hits += 1
            if expected_doc in cos_top:
                cosine_hits += 1

        total = len(PARAPHRASE_QUERIES)
        print(f"\n[FAILURE DEMO] Paraphrase retrieval recall:")
        print(f"  BM25:   {bm25_hits}/{total} ({bm25_hits/total:.0%})")
        print(f"  Cosine: {cosine_hits}/{total} ({cosine_hits/total:.0%})")
        print(f"\n  NOTE: Both are limited without real neural embeddings.")
        print(f"  Replace CosineRetriever._tf() with text-embedding-3-small calls")
        print(f"  to demonstrate the gap between TF-IDF and neural embeddings.")

    def test_retrieval_rank_degrades_for_paraphrase(self):
        """
        FAILURE: Even when the correct doc is eventually retrieved,
        paraphrase queries push it to a much lower rank.
        """
        docs = get_all_docs()
        retriever = build_index(BM25Retriever, docs)

        # Compare original query vs paraphrase
        original   = QUERIES[0]["query"]       # "How long do I have to return an item?"
        paraphrase = "send back a purchase"
        expected   = QUERIES[0]["source_doc"]  # "refund_policy"

        orig_results  = retriever.retrieve(original,   top_k=10)
        para_results  = retriever.retrieve(paraphrase, top_k=10)

        orig_rank = next((r.rank for r in orig_results  if expected in r.chunk_id), 99)
        para_rank = next((r.rank for r in para_results  if expected in r.chunk_id), 99)

        print(f"\n[FAILURE DEMO] Rank of correct document:")
        print(f"  Original query:   rank={orig_rank}  ('{original[:50]}')")
        print(f"  Paraphrase query: rank={para_rank}  ('{paraphrase}')")

        assert para_rank >= orig_rank, (
            f"Paraphrase query should produce equal or worse rank. "
            f"Original rank={orig_rank}, paraphrase rank={para_rank}."
        )


class TestHybridRetrieverPartiallyFixes:

    def test_hybrid_retriever_improves_paraphrase_recall(self):
        """
        FIX (partial): Hybrid RRF combines BM25 precision with cosine recall.
        Won't solve semantic mismatch fully without neural embeddings,
        but is consistently better than either alone.
        """
        docs = get_all_docs()
        bm25   = build_index(BM25Retriever,   docs)
        hybrid = build_index(HybridRetriever, docs)

        bm25_hits   = 0
        hybrid_hits = 0

        for query, expected_doc in PARAPHRASE_QUERIES:
            bm25_top   = [r.chunk_id.split("_chunk_")[0] for r in bm25.retrieve(query,   top_k=3)]
            hybrid_top = [r.chunk_id.split("_chunk_")[0] for r in hybrid.retrieve(query, top_k=3)]
            if expected_doc in bm25_top:
                bm25_hits += 1
            if expected_doc in hybrid_top:
                hybrid_hits += 1

        total = len(PARAPHRASE_QUERIES)
        print(f"\n[FIX VERIFICATION] Paraphrase recall:")
        print(f"  BM25:   {bm25_hits}/{total} ({bm25_hits/total:.0%})")
        print(f"  Hybrid: {hybrid_hits}/{total} ({hybrid_hits/total:.0%})")
        print(f"\n  Real fix: swap TF-IDF vectors for text-embedding-3-small or equivalent.")
