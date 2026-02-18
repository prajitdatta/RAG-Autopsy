"""
failures/f01_chunking/test_chunk_too_large.py

FAILURE: Chunks are too large — entire documents become single chunks.

ROOT CAUSE:
    When chunk_size equals or exceeds document length, the entire document
    becomes one chunk. The LLM receives the full document as context for every
    query. This causes:
    1. Context window overflow for long documents
    2. Attention dilution — the answer is buried in irrelevant text
    3. Higher cost — you pay for every token even when 90% is irrelevant

IMPACT:
    Moderate for short docs. Catastrophic for long docs (legal contracts,
    handbooks, technical manuals). Models lose the answer in the noise.

RUN:
    pytest failures/f01_chunking/test_chunk_too_large.py -v
"""

import pytest
from src.chunker import Chunker
from src.rag_pipeline import RAGPipeline
from data.sample_docs import get_doc, get_all_docs
from data.sample_queries import get_query


CONTEXT_WINDOW_LIMIT_CHARS = 16000  # ~4000 tokens


class TestChunkTooLarge:

    def test_oversized_chunks_exceed_context_window(self):
        """
        FAILURE: chunk_size=10000 produces chunks that won't fit in a
        typical LLM context window when multiple are combined.
        """
        docs = get_all_docs()
        chunker = Chunker(strategy="fixed", chunk_size=10000, overlap=0)

        all_chunks = []
        for doc in docs:
            all_chunks.extend(chunker.chunk(doc["content"]))

        # Simulate retrieving top 5 chunks and assembling context
        top5_text = " ".join(c.text for c in all_chunks[:5])
        total_chars = len(top5_text)

        print(f"\n[FAILURE DEMO] Top-5 chunks assembled context size: {total_chars:,} chars")
        print(f"  Context window limit: {CONTEXT_WINDOW_LIMIT_CHARS:,} chars")
        print(f"  Overflow: {max(0, total_chars - CONTEXT_WINDOW_LIMIT_CHARS):,} chars")

        assert total_chars > CONTEXT_WINDOW_LIMIT_CHARS, (
            f"Expected context to overflow limit of {CONTEXT_WINDOW_LIMIT_CHARS:,} chars. "
            f"Got {total_chars:,} chars."
        )

    def test_single_chunk_contains_entire_document(self):
        """
        FAILURE: When chunk_size is large, the entire document is one chunk.
        This means irrelevant content is always sent as context.
        """
        doc_text = get_doc("employee_handbook")
        chunker = Chunker(strategy="fixed", chunk_size=len(doc_text) + 1, overlap=0)
        chunks = chunker.chunk(doc_text)

        print(f"\n[FAILURE DEMO] Document length: {len(doc_text):,} chars")
        print(f"  Chunks produced: {len(chunks)}")
        if chunks:
            print(f"  Single chunk size: {len(chunks[0].text):,} chars")

        assert len(chunks) == 1, (
            f"Expected 1 giant chunk, got {len(chunks)}. "
            "Large chunk_size means the whole document is one chunk."
        )

    def test_oversized_chunks_dilute_answer(self):
        """
        FAILURE: When the retrieved chunk is the entire document,
        the correct answer is diluted by ~10x more irrelevant text.
        """
        query_data = get_query("q04")  # "What rating for promotion?"
        doc_text = get_doc("employee_handbook")

        # Build oversized pipeline
        pipeline = RAGPipeline(
            chunker_strategy="fixed",
            chunk_size=len(doc_text),
            overlap=0,
            top_k=1,
            evaluate=True,
        )
        pipeline.index([{"id": "handbook", "text": doc_text}])
        result = pipeline.query(query_data["query"], key_facts=query_data["key_facts"])

        # Measure noise ratio: irrelevant context vs answer-containing context
        chunk_text = result.chunks_retrieved[0].text if result.chunks_retrieved else ""
        answer_fragment = "rating of 4"
        answer_start = chunk_text.lower().find(answer_fragment)
        irrelevant_chars = len(chunk_text) - len(answer_fragment) if answer_start >= 0 else len(chunk_text)
        noise_ratio = irrelevant_chars / len(chunk_text) if chunk_text else 1.0

        print(f"\n[FAILURE DEMO] Context noise ratio with oversized chunk: {noise_ratio:.0%}")
        print(f"  Total context: {len(chunk_text):,} chars")
        print(f"  Irrelevant: {irrelevant_chars:,} chars ({noise_ratio:.0%})")

        assert noise_ratio > 0.8, (
            f"Expected >80% noise in oversized chunk, got {noise_ratio:.0%}. "
            "The relevant answer is buried in irrelevant content."
        )

    def test_token_cost_wasted_with_oversized_chunks(self):
        """
        FAILURE: Oversized chunks waste tokens, increasing API cost.
        For every query, you pay for the entire document even if the
        answer is in 2% of it.
        """
        doc_text = get_doc("financial_report")
        answer_fragment = "€847.3 million"  # The actual answer to a revenue question

        oversized_chunker = Chunker(strategy="fixed", chunk_size=len(doc_text), overlap=0)
        optimal_chunker   = Chunker(strategy="recursive", chunk_size=400)

        oversized_chunks = oversized_chunker.chunk(doc_text)
        optimal_chunks   = optimal_chunker.chunk(doc_text)

        # Find the smallest chunk that still contains the answer
        def find_answer_chunk(chunks, answer):
            for c in chunks:
                if answer.lower() in c.text.lower():
                    return c
            return None

        oversized_answer_chunk = find_answer_chunk(oversized_chunks, answer_fragment)
        optimal_answer_chunk   = find_answer_chunk(optimal_chunks, answer_fragment)

        oversized_tokens = oversized_answer_chunk.token_estimate if oversized_answer_chunk else 0
        optimal_tokens   = optimal_answer_chunk.token_estimate if optimal_answer_chunk else 0

        if oversized_tokens > 0 and optimal_tokens > 0:
            waste_ratio = oversized_tokens / optimal_tokens
            print(f"\n[FAILURE DEMO] Token cost waste ratio: {waste_ratio:.1f}x")
            print(f"  Oversized chunk tokens: {oversized_tokens}")
            print(f"  Optimal chunk tokens:   {optimal_tokens}")
            assert waste_ratio > 3.0, (
                f"Expected ≥3x token waste with oversized chunks. Got {waste_ratio:.1f}x."
            )


class TestOptimalChunkSizeFixes:

    def test_recursive_chunking_fits_context_window(self):
        """FIX: Properly-sized chunks fit within the context window."""
        docs = get_all_docs()
        chunker = Chunker(strategy="recursive", chunk_size=400)

        all_chunks = []
        for doc in docs:
            all_chunks.extend(chunker.chunk(doc["content"]))

        # Even top 5 chunks should fit
        top5_text = " ".join(c.text for c in all_chunks[:5])
        total_chars = len(top5_text)

        print(f"\n[FIX VERIFICATION] Top-5 recursive chunks: {total_chars:,} chars")
        assert total_chars < CONTEXT_WINDOW_LIMIT_CHARS, (
            f"Recursive chunking should produce chunks that fit the context window. "
            f"Got {total_chars:,} > {CONTEXT_WINDOW_LIMIT_CHARS:,} chars."
        )

    def test_granular_chunks_produce_precise_retrieval(self):
        """FIX: Small, precise chunks retrieve only the relevant section."""
        query_data = get_query("q04")
        doc_text   = get_doc("employee_handbook")

        pipeline = RAGPipeline(
            chunker_strategy="recursive",
            chunk_size=400,
            top_k=3,
            evaluate=True,
        )
        pipeline.index([{"id": "handbook", "text": doc_text}])
        result = pipeline.query(query_data["query"], key_facts=query_data["key_facts"])

        # All retrieved chunks should be smaller than the full document
        for r in result.chunks_retrieved:
            assert len(r.text) < len(doc_text), (
                f"Retrieved chunk should be smaller than the full document. "
                f"Got {len(r.text)} chars vs doc {len(doc_text)} chars."
            )

        print(f"\n[FIX VERIFICATION] Key fact coverage: {result.evaluation.key_fact_coverage:.2f}")
