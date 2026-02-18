"""
failures/f06_production/test_no_fallback.py

FAILURE: RAG pipeline has no fallback when retrieval returns nothing useful.
         The LLM hallucinates an answer rather than saying "I don't know."

RUN:
    pytest failures/f06_production/test_no_fallback.py -v
"""

import pytest
from src.rag_pipeline import RAGPipeline
from src.generator import MockGenerator
from src.evaluator import Evaluator
from data.sample_docs import get_all_docs


OUT_OF_SCOPE_QUERIES = [
    "What is the weather forecast for Stockholm tomorrow?",
    "Who won the Champions League in 2019?",
    "What is the chemical formula for caffeine?",
    "How do I reset my Wi-Fi router?",
]


class TestNoFallback:

    def test_pipeline_answers_out_of_scope_queries(self):
        """
        FAILURE: When the query has nothing to do with indexed documents,
        the pipeline still generates an answer — from hallucinated context.
        """
        docs = get_all_docs()
        pipeline = RAGPipeline(
            chunker_strategy="recursive",
            chunk_size=400,
            top_k=3,
            generator_mode="hallucinating",
        )
        pipeline.index([{"id": d["id"], "text": d["content"]} for d in docs])

        print(f"\n[FAILURE DEMO] Out-of-scope queries with no fallback:")
        for query in OUT_OF_SCOPE_QUERIES:
            result = pipeline.query(query)
            print(f"\n  Query: '{query}'")
            print(f"  Top retrieved chunk: '{result.chunks_retrieved[0].text[:80]}...' (irrelevant)")
            print(f"  Response: '{result.generation.response[:100]}'")
            print(f"  ❌ Should have said: 'I cannot find this in the provided documents'")

    def test_fix_low_relevance_triggers_fallback(self):
        """
        FIX: Check retrieval relevance score. Below a threshold, refuse to answer.
        """
        from src.retriever import HybridRetriever

        docs = get_all_docs()
        retriever = HybridRetriever()
        retriever.index([{"id": d["id"], "text": d["content"]} for d in docs])

        RELEVANCE_THRESHOLD = 0.05

        print(f"\n[FIX VERIFICATION] Fallback on low relevance (threshold={RELEVANCE_THRESHOLD}):")
        for query in OUT_OF_SCOPE_QUERIES[:2]:
            results = retriever.retrieve(query, top_k=3)
            max_score = max(r.score for r in results) if results else 0

            if max_score < RELEVANCE_THRESHOLD:
                response = "I cannot find relevant information in the provided documents."
                action   = "REFUSED (correct)"
            else:
                response = "Would proceed with generation..."
                action   = "PROCEEDED"

            print(f"\n  Query:     '{query}'")
            print(f"  Max score: {max_score:.4f}")
            print(f"  Action:    {action}")
            print(f"  Response:  '{response}'")
