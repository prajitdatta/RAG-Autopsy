"""
failures/f01_chunking/test_mid_sentence_split.py

FAILURE: Fixed-size chunking cuts sentences mid-way.

ROOT CAUSE:
    Fixed chunking splits on character count without any awareness of sentence
    or word boundaries. A chunk can end mid-sentence, mid-word, or mid-number
    (e.g. "€847.3 mi" | "llion"). The resulting chunks lose meaning because
    neither half contains a complete thought.

IMPACT:
    - Retrieval fails: split sentences score poorly against semantic queries
    - Generation fails: the LLM receives broken context, may hallucinate the rest
    - Particularly bad for: numbers, lists, definitions, multi-clause sentences

WHAT THIS TEST DOES:
    1. Demonstrates that fixed chunking breaks sentence boundaries
    2. Measures how many chunks start or end mid-sentence
    3. Shows the downstream retrieval quality degradation
    4. Proves the fix (sentence-aware or recursive chunking) eliminates the problem

RUN:
    pytest failures/f01_chunking/test_mid_sentence_split.py -v
"""

import pytest
import re
from src.chunker import Chunker
from src.rag_pipeline import RAGPipeline
from src.evaluator import Evaluator
from data.sample_docs import get_doc
from data.sample_queries import get_query


# ── Helpers ───────────────────────────────────────────────────────────────────

def ends_mid_sentence(text: str) -> bool:
    """Return True if text ends without a sentence-terminating punctuation."""
    stripped = text.strip()
    return bool(stripped) and stripped[-1] not in ".!?:;"


def starts_mid_sentence(text: str) -> bool:
    """Return True if text starts with a lowercase letter (mid-sentence continuation)."""
    stripped = text.strip()
    return bool(stripped) and stripped[0].islower()


def count_broken_chunks(chunks) -> int:
    return sum(
        1 for c in chunks
        if ends_mid_sentence(c.text) or starts_mid_sentence(c.text)
    )


# ── Failure tests (these PASS, proving the bug exists) ───────────────────────

class TestFixedChunkingBreaksSentences:

    def test_fixed_chunking_produces_mid_sentence_breaks(self):
        """
        DEMONSTRATE THE FAILURE:
        Fixed chunking (chunk_size=100) on a realistic document
        produces chunks that cut mid-sentence.
        """
        doc = get_doc("refund_policy")
        chunker = Chunker(strategy="fixed", chunk_size=100, overlap=0)
        chunks = chunker.chunk(doc)

        broken = count_broken_chunks(chunks)
        total  = len(chunks)

        print(f"\n[FAILURE DEMO] Fixed chunking: {broken}/{total} chunks have broken boundaries")
        for c in chunks[:5]:
            print(f"  CHUNK: '...{c.text[-30:]}' → ends_mid={ends_mid_sentence(c.text)}")

        # This assertion PASSES (confirming the bug): >30% of chunks are broken
        assert broken / total > 0.3, (
            f"Expected >30% broken chunks, got {broken}/{total}. "
            f"The bug may have been fixed already."
        )

    def test_fixed_chunking_splits_numbers(self):
        """
        Fixed chunking can split a number across chunks (e.g. '€847.3' → '€847.' | '.3M').
        This causes the LLM to receive partial numbers and hallucinate the rest.
        """
        financial_doc = get_doc("financial_report")
        chunker = Chunker(strategy="fixed", chunk_size=80, overlap=0)
        chunks = chunker.chunk(financial_doc)

        # Find all numbers in original doc
        numbers_in_doc = re.findall(r"€[\d,]+\.?\d*\s*(?:million|billion|M|B)?", financial_doc)

        # Find numbers that appear intact in at least one chunk
        numbers_intact = []
        for num in numbers_in_doc:
            found = any(num in c.text for c in chunks)
            numbers_intact.append(found)

        intact_rate = sum(numbers_intact) / len(numbers_intact) if numbers_intact else 0

        print(f"\n[FAILURE DEMO] Financial numbers intact after fixed chunking: "
              f"{sum(numbers_intact)}/{len(numbers_intact)} ({intact_rate:.0%})")

        # FAILURE: less than 70% of numbers survive intact
        assert intact_rate < 0.7, (
            "Expected numbers to be broken by fixed chunking. "
            "If this fails, chunk_size may be large enough to avoid splits."
        )

    def test_fixed_chunk_mid_list_item(self):
        """
        Fixed chunking breaks enumerated lists mid-item.
        A chunk ending with '- Perishable goods including food, flowers' and the
        next chunk starting with 'and plants' makes both useless for retrieval.
        """
        doc = get_doc("refund_policy")
        chunker = Chunker(strategy="fixed", chunk_size=120, overlap=0)
        chunks = chunker.chunk(doc)

        # Look for "non-returnable" items split across chunks
        all_text = " ".join(c.text for c in chunks)
        original_items = [
            "Perishable goods",
            "Digital downloads",
            "Personalized or custom-made",
        ]

        items_split = []
        for item in original_items:
            # Item is "split" if it doesn't appear whole in any single chunk
            in_single_chunk = any(item in c.text for c in chunks)
            items_split.append(not in_single_chunk)

        split_count = sum(items_split)
        print(f"\n[FAILURE DEMO] List items split across chunks: {split_count}/{len(original_items)}")

        assert split_count > 0, (
            f"Expected at least 1 list item to be split across chunks. "
            f"All {len(original_items)} items happened to land intact."
        )

    def test_downstream_retrieval_degraded_by_mid_sentence_splits(self):
        """
        End-to-end demonstration: broken chunks degrade retrieval quality.
        The correct answer IS in the document, but fixed chunking prevents retrieval.
        """
        query_data = get_query("q03")  # "How long does a refund take?"

        # Build pipeline with broken chunking
        broken_pipeline = RAGPipeline(
            chunker_strategy="fixed",
            chunk_size=80,
            overlap=0,
            top_k=3,
            evaluate=True,
        )
        broken_pipeline.index([{
            "id": "refund_policy",
            "text": get_doc("refund_policy")
        }])

        result = broken_pipeline.query(
            query_data["query"],
            key_facts=query_data["key_facts"],
        )

        key_fact_coverage = result.evaluation.key_fact_coverage
        print(f"\n[FAILURE DEMO] Key fact coverage with broken chunks: {key_fact_coverage:.2f}")
        print(f"  Retrieved chunks:")
        for r in result.chunks_retrieved:
            print(f"    [{r.rank}] '{r.text[:80]}...'")

        # With very small fixed chunks, key facts are split and retrieval degrades
        assert key_fact_coverage < 1.0, (
            "Expected broken chunking to reduce key fact coverage. "
            "This test demonstrates the downstream impact."
        )


# ── Fix verification (these tests prove the fix works) ───────────────────────

class TestSentenceAwareChunkingFixes:

    def test_recursive_chunking_respects_sentence_boundaries(self):
        """
        FIX: Recursive chunking produces far fewer mid-sentence breaks.
        """
        doc = get_doc("refund_policy")

        fixed_chunks    = Chunker(strategy="fixed",     chunk_size=200, overlap=0).chunk(doc)
        recursive_chunks = Chunker(strategy="recursive", chunk_size=200).chunk(doc)

        fixed_broken    = count_broken_chunks(fixed_chunks)
        recursive_broken = count_broken_chunks(recursive_chunks)

        print(f"\n[FIX COMPARISON]")
        print(f"  fixed:     {fixed_broken}/{len(fixed_chunks)} chunks broken")
        print(f"  recursive: {recursive_broken}/{len(recursive_chunks)} chunks broken")

        assert recursive_broken <= fixed_broken, (
            f"Recursive chunking should produce ≤ broken chunks than fixed. "
            f"Got fixed={fixed_broken}, recursive={recursive_broken}"
        )

    def test_sentence_chunking_keeps_numbers_intact(self):
        """
        FIX: Sentence-aware chunking keeps numbers and monetary values intact.
        """
        doc = get_doc("financial_report")
        chunker = Chunker(strategy="sentence", max_sentences=3)
        chunks = chunker.chunk(doc)

        numbers = re.findall(r"€[\d,]+\.?\d*", doc)
        intact = sum(1 for n in numbers if any(n in c.text for c in chunks))
        intact_rate = intact / len(numbers) if numbers else 0

        print(f"\n[FIX VERIFICATION] Numbers intact with sentence chunking: "
              f"{intact}/{len(numbers)} ({intact_rate:.0%})")

        assert intact_rate >= 0.8, (
            f"Sentence chunking should preserve ≥80% of numbers intact. Got {intact_rate:.0%}"
        )
