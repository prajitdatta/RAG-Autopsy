"""
failures/f03_context_assembly/test_context_overflow.py

FAILURE: Too many chunks overflow the context window, causing truncation.
         The LLM silently receives an incomplete prompt and answers from partial context.

ROOT CAUSE:
    Passing 20+ chunks of 500 chars each = 10,000+ chars = ~2,500 tokens.
    With a system prompt and response budget, you hit the context limit.
    The provider silently truncates — usually from the END of the context,
    which happens to be where later, potentially more relevant chunks live.

RUN:
    pytest failures/f03_context_assembly/test_context_overflow.py -v
"""

import pytest
from src.rag_pipeline import RAGPipeline
from data.sample_docs import get_all_docs, get_doc
from data.sample_queries import get_query


# Simulate a 4000-token (~16000 char) context window
CONTEXT_WINDOW_CHARS = 16000
# Reserve for system prompt + response
RESERVED_CHARS = 4000
AVAILABLE_FOR_CONTEXT = CONTEXT_WINDOW_CHARS - RESERVED_CHARS


class TestContextOverflow:

    def test_too_many_chunks_exceed_context_window(self):
        """
        FAILURE: Retrieving k=20 chunks with large chunk sizes overflows the context window.
        """
        docs = get_all_docs()

        pipeline = RAGPipeline(
            chunker_strategy="fixed",
            chunk_size=600,
            overlap=0,
            top_k=20,
        )
        pipeline.index([{"id": d["id"], "text": d["content"]} for d in docs])
        result = pipeline.query("What are the refund processing times?")

        total_context_chars = sum(len(r.text) for r in result.chunks_retrieved)
        overflows = total_context_chars > AVAILABLE_FOR_CONTEXT

        print(f"\n[FAILURE DEMO] Context assembly with k=20, chunk_size=600:")
        print(f"  Total context: {total_context_chars:,} chars")
        print(f"  Available:     {AVAILABLE_FOR_CONTEXT:,} chars")
        print(f"  Overflow:      {max(0, total_context_chars - AVAILABLE_FOR_CONTEXT):,} chars")
        print(f"  Chunks lost to truncation: ~{max(0, total_context_chars - AVAILABLE_FOR_CONTEXT) // 600}")

        if overflows:
            print(f"\n  SILENT TRUNCATION: LLM receives incomplete context with no error.")

    def test_truncation_removes_later_chunks(self):
        """
        FAILURE: Truncation removes the LAST chunks in the context.
        If the most relevant chunk was ranked 15th (near the end), it's gone.
        """
        docs = get_all_docs()

        pipeline = RAGPipeline(
            chunker_strategy="fixed",
            chunk_size=500,
            overlap=0,
            top_k=20,
        )
        pipeline.index([{"id": d["id"], "text": d["content"]} for d in docs])
        result = pipeline.query("What is the Metformin starting dose?")

        # Simulate what the LLM actually receives after truncation
        all_context_text = "\n---\n".join(r.text for r in result.chunks_retrieved)
        truncated         = all_context_text[:AVAILABLE_FOR_CONTEXT]

        # Find where the answer is
        answer_fragment = "500 mg"
        in_full    = answer_fragment in all_context_text
        in_truncated = answer_fragment in truncated

        print(f"\n[FAILURE DEMO] Answer '{answer_fragment}':")
        print(f"  In full context (k=20):     {in_full}")
        print(f"  In truncated context:       {in_truncated}")
        if in_full and not in_truncated:
            print(f"  ❌ LOST TO TRUNCATION — answer retrieved but then cut off")


class TestContextOverflowFix:

    def test_fix_budget_context_size(self):
        """
        FIX: Enforce a context budget. Stop adding chunks once the budget is reached.
        """
        docs = get_all_docs()

        pipeline = RAGPipeline(
            chunker_strategy="recursive",
            chunk_size=400,
            top_k=5,  # Conservative k
        )
        pipeline.index([{"id": d["id"], "text": d["content"]} for d in docs])
        result = pipeline.query("What are the refund processing times?")

        total_context_chars = sum(len(r.text) for r in result.chunks_retrieved)
        fits = total_context_chars <= AVAILABLE_FOR_CONTEXT

        print(f"\n[FIX VERIFICATION] Context with k=5, chunk_size=400:")
        print(f"  Total context: {total_context_chars:,} chars")
        print(f"  Fits in window: {fits}")
        assert fits, f"Context should fit in window. Got {total_context_chars} > {AVAILABLE_FOR_CONTEXT}"
