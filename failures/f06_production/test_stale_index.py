"""
failures/f06_production/test_stale_index.py

FAILURE: The vector index is not updated when source documents change.
         Users receive answers based on outdated policy.

ROOT CAUSE:
    A common deployment pattern: index built once, served forever.
    When the underlying documents change (new policy, updated price, corrected
    guideline), the index still contains old vectors pointing to old text.
    The retriever confidently returns the old chunk. The LLM confidently
    answers from it. The user gets the wrong, outdated answer.

IMPACT:
    Silent correctness regression. Particularly damaging for:
    - Legal/compliance documents (policy changes)
    - Medical guidelines (updated protocols)
    - Pricing (rate changes)
    - Financial data (quarterly updates)

RUN:
    pytest failures/f06_production/test_stale_index.py -v
"""

import pytest
from src.rag_pipeline import RAGPipeline
from src.evaluator import Evaluator


# ── Simulate document version changes ─────────────────────────────────────────

POLICY_V1 = {
    "id": "refund_policy",
    "text": (
        "Return and Refund Policy — Version 1.0\n"
        "Customers may return items within 30 days of purchase for a full refund.\n"
        "Refunds are processed within 5-7 business days.\n"
        "Items must be in original condition with proof of purchase.\n"
        "Non-returnable items include digital downloads and perishable goods."
    ),
}

POLICY_V2 = {
    "id": "refund_policy",
    "text": (
        "Return and Refund Policy — Version 2.0 (Updated March 2025)\n"
        "IMPORTANT CHANGE: The return window has been EXTENDED to 60 days.\n"
        "Refunds are now processed within 3-5 business days (previously 5-7).\n"
        "Items must be in original condition with proof of purchase.\n"
        "Non-returnable items now also include assembled furniture.\n"
        "New: Free return shipping on all orders over €50."
    ),
}


class TestStaleIndex:

    def test_stale_index_returns_old_policy(self):
        """
        FAILURE: Index built from V1 returns V1 content even after V2 is published.
        """
        # Build index from V1
        pipeline = RAGPipeline(chunker_strategy="recursive", chunk_size=300, top_k=3)
        pipeline.index([POLICY_V1])

        # Query after V2 is live (but index NOT updated)
        result = pipeline.query("What is the return window?")
        response_text = " ".join(r.text for r in result.chunks_retrieved)

        old_info_present = "30 days" in response_text
        new_info_present = "60 days" in response_text

        print(f"\n[FAILURE DEMO] Stale index after V2 policy update:")
        print(f"  Old (V1) content present: {old_info_present} ('30 days')")
        print(f"  New (V2) content present: {new_info_present} ('60 days')")

        assert old_info_present, "Stale index should return old 30-day policy"
        assert not new_info_present, "Stale index should NOT return new 60-day policy"

    def test_updated_index_returns_new_policy(self):
        """
        FIX: Rebuilding the index after document update returns correct content.
        """
        pipeline = RAGPipeline(chunker_strategy="recursive", chunk_size=300, top_k=3)

        # Update: rebuild with V2
        pipeline.index([POLICY_V2])
        result = pipeline.query("What is the return window?")
        response_text = " ".join(r.text for r in result.chunks_retrieved)

        new_info_present = "60 days" in response_text
        print(f"\n[FIX VERIFICATION] After index rebuild with V2:")
        print(f"  New content (60 days) present: {new_info_present}")
        assert new_info_present, "Rebuilt index should contain new 60-day return window"

    def test_stale_index_impacts_answer_correctness(self):
        """
        End-to-end: stale index produces wrong answer; updated index produces correct answer.
        """
        evaluator = Evaluator()
        query = "What is the current return window?"
        key_facts_v2 = ["60 days"]

        # Stale index (V1)
        stale_pipeline = RAGPipeline(chunker_strategy="recursive", chunk_size=300, top_k=3)
        stale_pipeline.index([POLICY_V1])
        stale_result = stale_pipeline.query(query, key_facts=key_facts_v2)

        # Fresh index (V2)
        fresh_pipeline = RAGPipeline(chunker_strategy="recursive", chunk_size=300, top_k=3)
        fresh_pipeline.index([POLICY_V2])
        fresh_result = fresh_pipeline.query(query, key_facts=key_facts_v2)

        stale_eval = evaluator.evaluate(query, stale_result.generation.response,
                                        [POLICY_V1["text"]], key_facts=key_facts_v2)
        fresh_eval = evaluator.evaluate(query, fresh_result.generation.response,
                                        [POLICY_V2["text"]], key_facts=key_facts_v2)

        print(f"\n[FAILURE DEMO] Answer correctness (key fact: '60 days'):")
        print(f"  Stale index: coverage={stale_eval.key_fact_coverage:.2f}, "
              f"response='{stale_result.generation.response[:80]}'")
        print(f"  Fresh index: coverage={fresh_eval.key_fact_coverage:.2f}, "
              f"response='{fresh_result.generation.response[:80]}'")
