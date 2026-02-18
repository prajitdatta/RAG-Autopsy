"""
failures/f01_chunking/test_chunk_too_small.py

FAILURE: Chunks are too small — each chunk loses necessary context.

ROOT CAUSE:
    When chunk_size is very small (e.g. 50 chars), each chunk becomes a
    sentence fragment. Individual fragments look locally relevant but miss
    the surrounding context needed to answer the question fully.

    Example: "Approved refunds are processed within 5-7 business days."
    → Split into: "Approved refunds are" | "processed within 5-7" | "business days."
    Each fragment separately scores low on retrieval. The assembled answer
    is missing critical parts of the original sentence.

IMPACT:
    - Retrieved chunks answer PART of the question but miss key qualifiers
    - Multi-part answers require combining 10+ tiny chunks
    - Precision collapses: top-k retrieval fills slots with redundant micro-fragments

RUN:
    pytest failures/f01_chunking/test_chunk_too_small.py -v
"""

import pytest
from src.chunker import Chunker
from src.rag_pipeline import RAGPipeline
from data.sample_docs import get_doc
from data.sample_queries import get_query


MINIMUM_USEFUL_CHUNK_CHARS = 100  # Below this, chunks are likely too fragmented


class TestChunkTooSmall:

    def test_tiny_chunks_lose_context(self):
        """
        FAILURE: 50-char chunks fragment complete thoughts into useless pieces.
        """
        doc = get_doc("refund_policy")
        chunker = Chunker(strategy="fixed", chunk_size=50, overlap=0)
        chunks = chunker.chunk(doc)

        too_small = [c for c in chunks if len(c.text) < MINIMUM_USEFUL_CHUNK_CHARS]
        fraction_too_small = len(too_small) / len(chunks)

        print(f"\n[FAILURE DEMO] Chunks smaller than {MINIMUM_USEFUL_CHUNK_CHARS} chars: "
              f"{len(too_small)}/{len(chunks)} ({fraction_too_small:.0%})")
        print(f"  Examples of tiny chunks:")
        for c in too_small[:5]:
            print(f"    '{c.text}'")

        assert fraction_too_small > 0.5, (
            f"Expected >50% of chunks to be too small. Got {fraction_too_small:.0%}."
        )

    def test_tiny_chunks_produce_redundant_retrievals(self):
        """
        FAILURE: Top-k retrieval with tiny chunks returns fragments from the
        same sentence, wasting all k slots on redundant information.
        """
        doc = get_doc("refund_policy")
        query_data = get_query("q03")  # "How long does a refund take?"

        pipeline = RAGPipeline(
            chunker_strategy="fixed",
            chunk_size=60,
            overlap=10,
            top_k=5,
        )
        pipeline.index([{"id": "refund_policy", "text": doc}])
        result = pipeline.query(query_data["query"])

        # Check for redundancy: multiple retrieved chunks containing the same keyword
        retrieved_texts = [r.text for r in result.chunks_retrieved]
        keyword = "business days"
        overlapping = sum(1 for t in retrieved_texts if keyword in t.lower())

        print(f"\n[FAILURE DEMO] '{keyword}' appears in {overlapping}/{len(retrieved_texts)} retrieved chunks")
        print(f"  This is redundant — multiple slots wasted on the same sentence fragment")
        for r in result.chunks_retrieved:
            print(f"  Rank {r.rank}: '{r.text[:70]}'")

    def test_complete_answer_requires_too_many_tiny_chunks(self):
        """
        FAILURE: To get the complete answer with tiny chunks, you'd need k=15+,
        which defeats the purpose of retrieval and adds noise.
        """
        doc    = get_doc("refund_policy")
        answer = "5-7 business days"

        for chunk_size in [50, 100, 200, 400]:
            chunker = Chunker(strategy="fixed", chunk_size=chunk_size, overlap=0)
            chunks = chunker.chunk(doc)
            # Find how many chunks contain the answer fragment
            containing = [c for c in chunks if "5-7" in c.text or "business days" in c.text]
            complete   = [c for c in chunks if "5-7 business days" in c.text]

            print(f"\n  chunk_size={chunk_size:4d}: "
                  f"partial_matches={len(containing)}, "
                  f"complete_phrase={len(complete)}, "
                  f"total_chunks={len(chunks)}")

        # With size=50: answer is split, complete phrase appears in 0 chunks
        small_chunker = Chunker(strategy="fixed", chunk_size=50, overlap=0)
        small_chunks  = small_chunker.chunk(doc)
        complete_in_small = sum(1 for c in small_chunks if "5-7 business days" in c.text)

        assert complete_in_small == 0, (
            f"Expected tiny chunks to split '5-7 business days' across boundaries. "
            f"Found {complete_in_small} complete occurrences."
        )

    def test_tiny_chunks_inflate_index_size(self):
        """
        FAILURE: Tiny chunks create 10x more index entries than necessary,
        increasing memory usage, index build time, and retrieval latency.
        """
        doc = get_doc("employee_handbook")
        small_chunker    = Chunker(strategy="fixed",     chunk_size=60,  overlap=0)
        optimal_chunker  = Chunker(strategy="recursive", chunk_size=400)

        small_chunks   = small_chunker.chunk(doc)
        optimal_chunks = optimal_chunker.chunk(doc)

        ratio = len(small_chunks) / len(optimal_chunks)
        print(f"\n[FAILURE DEMO] Index inflation: {len(small_chunks)} tiny chunks "
              f"vs {len(optimal_chunks)} optimal chunks ({ratio:.1f}x inflation)")

        assert len(small_chunks) > len(optimal_chunks) * 3, (
            f"Expected ≥3x more chunks with tiny size. "
            f"Got {len(small_chunks)} vs {len(optimal_chunks)}."
        )


class TestOptimalChunkSizeFixes:

    def test_sentence_chunking_produces_complete_answers(self):
        """FIX: Sentence chunking keeps complete phrases together."""
        doc = get_doc("refund_policy")
        chunker = Chunker(strategy="sentence", max_sentences=3)
        chunks = chunker.chunk(doc)

        complete = [c for c in chunks if "5-7 business days" in c.text]
        print(f"\n[FIX VERIFICATION] '5-7 business days' intact in "
              f"{len(complete)}/{len(chunks)} chunks")

        assert len(complete) >= 1, (
            "Sentence chunking should preserve '5-7 business days' intact in at least one chunk."
        )

    def test_recursive_chunks_have_meaningful_size(self):
        """FIX: Recursive chunks are all above the minimum useful size."""
        doc = get_doc("employee_handbook")
        chunker = Chunker(strategy="recursive", chunk_size=300)
        chunks = chunker.chunk(doc)

        too_small = [c for c in chunks if len(c.text) < MINIMUM_USEFUL_CHUNK_CHARS]
        fraction  = len(too_small) / len(chunks) if chunks else 0

        print(f"\n[FIX VERIFICATION] Chunks below min size: {len(too_small)}/{len(chunks)} ({fraction:.0%})")
        assert fraction < 0.1, (
            f"Recursive chunking should produce <10% too-small chunks. Got {fraction:.0%}."
        )
