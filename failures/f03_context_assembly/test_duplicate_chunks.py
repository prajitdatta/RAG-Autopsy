"""
failures/f03_context_assembly/test_duplicate_chunks.py

FAILURE: Retrieved context contains duplicate or near-duplicate chunks,
         wasting context space and biasing the LLM toward repeated content.

ROOT CAUSE:
    Duplicate chunks arise from three sources:
    1. Overlapping fixed-size chunks (overlap=64 means adjacent chunks share 64 chars)
    2. The same source paragraph appears in multiple indexed documents
    3. The retriever returns both "income: 55000" and "income: 55000.00" as different chunks

    When 3 out of 5 retrieved chunks are near-duplicates, the LLM:
    a) Wastes ~60% of its context budget on redundant text
    b) Anchors on the repeated content, even if wrong
    c) Effectively only has k=2 distinct pieces of information

RUN:
    pytest failures/f03_context_assembly/test_duplicate_chunks.py -v
"""

import pytest
import re
from src.chunker import Chunker
from src.retriever import HybridRetriever
from data.sample_docs import get_doc
from data.sample_queries import get_query


def jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    ta = set(re.findall(r"\b[a-z]{3,}\b", a.lower()))
    tb = set(re.findall(r"\b[a-z]{3,}\b", b.lower()))
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


def find_duplicate_pairs(chunks: list[str], threshold: float = 0.7) -> list[tuple[int, int, float]]:
    """Find all pairs of chunks with similarity above threshold."""
    pairs = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            sim = jaccard_similarity(chunks[i], chunks[j])
            if sim >= threshold:
                pairs.append((i, j, sim))
    return pairs


class TestDuplicateChunks:

    def test_overlapping_fixed_chunks_produce_near_duplicates(self):
        """
        FAILURE: Fixed chunking with overlap produces chunks that are
        60-80% identical to their neighbours. Retrieving adjacent chunks
        is nearly useless — you get the same information twice.
        """
        doc = get_doc("refund_policy")
        chunker = Chunker(strategy="fixed", chunk_size=200, overlap=100)  # 50% overlap
        chunks = chunker.chunk(doc)
        texts = [c.text for c in chunks]

        # Check similarity between adjacent chunks
        adjacent_sims = [
            jaccard_similarity(texts[i], texts[i + 1])
            for i in range(len(texts) - 1)
        ]

        high_overlap = [s for s in adjacent_sims if s > 0.5]
        avg_adjacent_sim = sum(adjacent_sims) / len(adjacent_sims) if adjacent_sims else 0

        print(f"\n[FAILURE DEMO] Fixed chunking with 50% overlap:")
        print(f"  Chunks produced: {len(chunks)}")
        print(f"  Average adjacent similarity: {avg_adjacent_sim:.2f}")
        print(f"  Adjacent chunk pairs with >50% overlap: {len(high_overlap)}/{len(adjacent_sims)}")

        assert avg_adjacent_sim > 0.3, (
            f"Adjacent fixed chunks should share significant content due to overlap. "
            f"Got avg_sim={avg_adjacent_sim:.2f}."
        )

    def test_retrieval_returns_duplicate_chunks(self):
        """
        FAILURE: Retriever returns near-duplicate chunks in top-k results,
        wasting context slots with repeated information.
        """
        doc = get_doc("refund_policy")
        # Index with overlapping chunks
        chunker = Chunker(strategy="fixed", chunk_size=180, overlap=90)
        chunks = chunker.chunk(doc)
        docs = [{"id": f"chunk_{c.index}", "text": c.text} for c in chunks]

        retriever = HybridRetriever()
        retriever.index(docs)

        query_data = get_query("q03")
        results = retriever.retrieve(query_data["query"], top_k=5)
        retrieved_texts = [r.text for r in results]

        duplicates = find_duplicate_pairs(retrieved_texts, threshold=0.5)

        print(f"\n[FAILURE DEMO] Duplicate chunks in top-5 retrieval:")
        print(f"  Retrieved {len(retrieved_texts)} chunks")
        print(f"  Near-duplicate pairs (sim>0.5): {len(duplicates)}")
        for i, j, sim in duplicates:
            print(f"    Chunk {i} ↔ Chunk {j}: {sim:.2f} similarity")
            print(f"      [{i}]: '{retrieved_texts[i][:60]}...'")
            print(f"      [{j}]: '{retrieved_texts[j][:60]}...'")

    def test_unique_content_fraction_of_context(self):
        """
        FAILURE: When duplicates are present, only a fraction of the context
        is unique information. Measure the actual information density.
        """
        doc = get_doc("employee_handbook")
        chunker = Chunker(strategy="fixed", chunk_size=150, overlap=75)
        chunks = chunker.chunk(doc)
        docs = [{"id": f"chunk_{c.index}", "text": c.text} for c in chunks]

        retriever = HybridRetriever()
        retriever.index(docs)
        results = retriever.retrieve("What are the promotion criteria?", top_k=5)

        # Measure unique tokens across all retrieved chunks
        all_text = " ".join(r.text for r in results)
        total_tokens = set(re.findall(r"\b[a-z]{3,}\b", all_text.lower()))

        # Tokens per individual chunk
        per_chunk_unique = [
            set(re.findall(r"\b[a-z]{3,}\b", r.text.lower()))
            for r in results
        ]

        # How many new tokens does each additional chunk add?
        seen = set()
        marginal_new = []
        for chunk_tokens in per_chunk_unique:
            new = chunk_tokens - seen
            marginal_new.append(len(new))
            seen |= chunk_tokens

        print(f"\n[FAILURE DEMO] Information density (new unique tokens per chunk):")
        for i, new in enumerate(marginal_new):
            bar = "█" * (new // 5)
            print(f"  Chunk {i+1}: {new:3d} new tokens  {bar}")

        # With duplicates, later chunks add fewer new tokens
        if len(marginal_new) >= 2:
            print(f"\n  Chunk 1 contributed {marginal_new[0]} unique tokens")
            print(f"  Chunk 5 contributed {marginal_new[-1]} unique tokens")
            print(f"  Information yield dropped by "
                  f"{(1 - marginal_new[-1]/marginal_new[0])*100:.0f}%")


class TestDuplicateChunkFixes:

    def test_fix_deduplicate_retrieved_chunks(self):
        """
        FIX: Remove near-duplicate chunks from retrieved results before
        passing to the LLM. Simple Jaccard deduplication.
        """
        doc = get_doc("refund_policy")
        chunker = Chunker(strategy="fixed", chunk_size=180, overlap=90)
        chunks = chunker.chunk(doc)
        docs = [{"id": f"chunk_{c.index}", "text": c.text} for c in chunks]

        retriever = HybridRetriever()
        retriever.index(docs)

        query_data = get_query("q03")
        raw_results = retriever.retrieve(query_data["query"], top_k=8)

        # Deduplication
        def deduplicate(results, threshold=0.6):
            kept = []
            for candidate in results:
                is_dup = any(
                    jaccard_similarity(candidate.text, kept_r.text) >= threshold
                    for kept_r in kept
                )
                if not is_dup:
                    kept.append(candidate)
            return kept

        deduped = deduplicate(raw_results)

        print(f"\n[FIX VERIFICATION] Deduplication:")
        print(f"  Before: {len(raw_results)} chunks")
        print(f"  After:  {len(deduped)} chunks")
        print(f"  Removed: {len(raw_results) - len(deduped)} duplicates")

        remaining_dups = find_duplicate_pairs([r.text for r in deduped], threshold=0.6)
        assert len(remaining_dups) == 0, (
            f"After deduplication, no near-duplicate pairs should remain. "
            f"Found {len(remaining_dups)}."
        )

    def test_fix_recursive_chunking_produces_fewer_duplicates(self):
        """
        FIX: Recursive chunking without overlap produces naturally distinct chunks.
        """
        doc = get_doc("refund_policy")

        overlapping_chunker = Chunker(strategy="fixed",     chunk_size=180, overlap=90)
        distinct_chunker    = Chunker(strategy="recursive", chunk_size=300)

        overlapping_chunks = overlapping_chunker.chunk(doc)
        distinct_chunks    = distinct_chunker.chunk(doc)

        overlap_texts   = [c.text for c in overlapping_chunks]
        distinct_texts  = [c.text for c in distinct_chunks]

        overlap_dups = find_duplicate_pairs(overlap_texts,  threshold=0.5)
        distinct_dups = find_duplicate_pairs(distinct_texts, threshold=0.5)

        print(f"\n[FIX VERIFICATION] Duplicate pairs:")
        print(f"  Fixed + overlap:   {len(overlap_dups)} pairs in {len(overlap_texts)} chunks")
        print(f"  Recursive:         {len(distinct_dups)} pairs in {len(distinct_texts)} chunks")

        assert len(distinct_dups) <= len(overlap_dups), (
            "Recursive chunking should produce fewer near-duplicate pairs."
        )
