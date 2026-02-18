"""
failures/f02_retrieval/test_wrong_top_k.py

FAILURE: top_k is set too low (misses the answer) or too high (adds noise).

ROOT CAUSE:
    top_k is usually hard-coded without understanding the data. Two opposing
    failure modes:

    LOW top_k (k=1 or k=2):
        The correct chunk may not be ranked #1. A slightly different phrasing,
        a synonym, or a vocabulary mismatch pushes the right chunk to rank 3-5.
        With k=1 you miss it entirely.

    HIGH top_k (k=20+):
        Every retrieved chunk is stuffed into the context. The correct chunk
        gets drowned out by irrelevant chunks from other documents. The LLM
        either picks the wrong answer or gets confused by contradictions.

IMPACT:
    Low k → recall failure. High k → precision failure.
    Both result in wrong or incomplete answers.

RUN:
    pytest failures/f02_retrieval/test_wrong_top_k.py -v
"""

import pytest
from src.rag_pipeline import RAGPipeline
from src.evaluator import Evaluator
from data.sample_docs import get_all_docs, get_doc
from data.sample_queries import QUERIES, get_query


class TestTopKTooLow:

    def test_k1_misses_answer_when_not_ranked_first(self):
        """
        FAILURE: With k=1, if the correct chunk is ranked #2 or lower, the
        pipeline has no answer and the LLM must hallucinate or refuse.
        """
        docs = get_all_docs()
        evaluator = Evaluator()

        results_k1 = []
        results_k5 = []

        for query_data in QUERIES[:5]:
            for k, results_list in [(1, results_k1), (5, results_k5)]:
                pipeline = RAGPipeline(
                    chunker_strategy="recursive",
                    chunk_size=400,
                    top_k=k,
                    evaluate=True,
                )
                pipeline.index([{
                    "id": d["id"], "text": d["content"]
                } for d in docs])

                result = pipeline.query(
                    query_data["query"],
                    key_facts=query_data["key_facts"],
                )
                results_list.append(result.evaluation.key_fact_coverage)

        avg_k1 = sum(results_k1) / len(results_k1)
        avg_k5 = sum(results_k5) / len(results_k5)

        print(f"\n[FAILURE DEMO] Average key fact coverage:")
        print(f"  k=1: {avg_k1:.2f}")
        print(f"  k=5: {avg_k5:.2f}")
        print(f"  Improvement from k=1 to k=5: {avg_k5 - avg_k1:+.2f}")

        assert avg_k5 >= avg_k1, (
            "Higher k should produce equal or better coverage than k=1."
        )

    def test_k1_pipeline_misses_multi_part_answers(self):
        """
        FAILURE: Some answers span multiple chunks. k=1 can only retrieve one.
        """
        # Query requires combining info from two sections of the employee handbook
        query = "What are the promotion criteria and what salary increase comes with a rating of 5?"
        key_facts = ["rating of 4", "18 months", "7-10%", "spot bonuses"]

        doc = get_doc("employee_handbook")
        pipeline_k1 = RAGPipeline(
            chunker_strategy="recursive", chunk_size=300, top_k=1, evaluate=True
        )
        pipeline_k1.index([{"id": "handbook", "text": doc}])
        result_k1 = pipeline_k1.query(query, key_facts=key_facts)

        pipeline_k5 = RAGPipeline(
            chunker_strategy="recursive", chunk_size=300, top_k=5, evaluate=True
        )
        pipeline_k5.index([{"id": "handbook", "text": doc}])
        result_k5 = pipeline_k5.query(query, key_facts=key_facts)

        print(f"\n[FAILURE DEMO] Multi-part answer coverage:")
        print(f"  k=1 coverage: {result_k1.evaluation.key_fact_coverage:.2f}")
        print(f"  k=5 coverage: {result_k5.evaluation.key_fact_coverage:.2f}")

        assert result_k1.evaluation.key_fact_coverage <= result_k5.evaluation.key_fact_coverage, (
            "k=5 should retrieve more relevant content for multi-part questions."
        )


class TestTopKTooHigh:

    def test_high_k_pollutes_context_with_irrelevant_docs(self):
        """
        FAILURE: With k=20 and a multi-document corpus, retrieved context
        contains chunks from completely different documents.
        """
        docs = get_all_docs()
        query_data = get_query("q01")  # Refund question → should retrieve refund_policy only

        pipeline = RAGPipeline(
            chunker_strategy="recursive",
            chunk_size=300,
            top_k=20,
        )
        pipeline.index([{"id": d["id"], "text": d["content"]} for d in docs])
        result = pipeline.query(query_data["query"])

        # Check how many retrieved chunks are from non-relevant documents
        irrelevant_sources = [
            r for r in result.chunks_retrieved
            if "refund" not in r.chunk_id and "financial" not in r.chunk_id
        ]

        irrelevant_fraction = len(irrelevant_sources) / len(result.chunks_retrieved)
        print(f"\n[FAILURE DEMO] k=20 retrieval for refund query:")
        print(f"  Total retrieved: {len(result.chunks_retrieved)}")
        print(f"  Irrelevant (from other docs): {len(irrelevant_sources)} ({irrelevant_fraction:.0%})")
        for r in irrelevant_sources[:3]:
            print(f"    Rank {r.rank}: [{r.chunk_id}] '{r.text[:60]}...'")

        assert irrelevant_fraction > 0.3, (
            f"Expected >30% irrelevant chunks with k=20. Got {irrelevant_fraction:.0%}. "
            "The corpus may be too small to demonstrate this failure mode clearly."
        )

    def test_high_k_increases_cost_without_quality_gain(self):
        """
        FAILURE: k=20 costs 4x more tokens than k=5 without meaningful quality gain.
        """
        docs = get_all_docs()
        query_data = get_query("q05")

        results = {}
        for k in [3, 5, 10, 20]:
            pipeline = RAGPipeline(
                chunker_strategy="recursive",
                chunk_size=300,
                top_k=k,
                evaluate=True,
            )
            pipeline.index([{"id": d["id"], "text": d["content"]} for d in docs])
            result = pipeline.query(query_data["query"], key_facts=query_data["key_facts"])

            context_chars = sum(len(r.text) for r in result.chunks_retrieved)
            results[k] = {
                "coverage": result.evaluation.key_fact_coverage,
                "context_chars": context_chars,
                "coverage_per_1k_chars": result.evaluation.key_fact_coverage / (context_chars / 1000),
            }

        print(f"\n[FAILURE DEMO] k vs quality/cost tradeoff:")
        print(f"  {'k':>4} | {'coverage':>10} | {'context_chars':>14} | {'coverage/1k_chars':>18}")
        print(f"  {'-'*55}")
        for k, r in results.items():
            print(f"  {k:>4} | {r['coverage']:>10.2f} | {r['context_chars']:>14,} | {r['coverage_per_1k_chars']:>18.3f}")

        # k=20 should cost more (more context chars) but not proportionally better
        cost_ratio = results[20]["context_chars"] / results[5]["context_chars"]
        quality_ratio = (results[20]["coverage"] + 0.001) / (results[5]["coverage"] + 0.001)

        assert cost_ratio > quality_ratio, (
            f"k=20 should cost more than the quality gain justifies. "
            f"cost_ratio={cost_ratio:.2f}, quality_ratio={quality_ratio:.2f}"
        )


class TestOptimalKFixes:

    def test_optimal_k_balances_recall_and_precision(self):
        """
        FIX: k=5 with good chunking typically balances recall and precision.
        Run a sweep and print the optimal k for this corpus.
        """
        docs = get_all_docs()
        coverages = {}

        for k in [1, 2, 3, 5, 8, 10]:
            total_coverage = 0
            for query_data in QUERIES[:5]:
                pipeline = RAGPipeline(
                    chunker_strategy="recursive",
                    chunk_size=400,
                    top_k=k,
                    evaluate=True,
                )
                pipeline.index([{"id": d["id"], "text": d["content"]} for d in docs])
                result = pipeline.query(
                    query_data["query"],
                    key_facts=query_data["key_facts"],
                )
                total_coverage += result.evaluation.key_fact_coverage
            coverages[k] = total_coverage / 5

        print(f"\n[FIX VERIFICATION] Coverage by k:")
        for k, cov in coverages.items():
            bar = "█" * int(cov * 20)
            print(f"  k={k:>2}: {cov:.2f} {bar}")

        best_k = max(coverages, key=coverages.get)
        print(f"\n  Optimal k for this corpus: {best_k}")

        # At least some k should do well
        assert max(coverages.values()) > 0.3, (
            "At least one k value should achieve >30% key fact coverage."
        )
