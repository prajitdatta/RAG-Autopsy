"""
failures/f05_evaluation/test_missing_groundedness_check.py

FAILURE: Evaluation only checks answer quality but not whether the answer
         is actually grounded in the retrieved context.

RUN:
    pytest failures/f05_evaluation/test_missing_groundedness_check.py -v
"""

import pytest
from src.evaluator import Evaluator
from data.sample_docs import get_doc


class TestMissingGroundednessCheck:

    def test_fluent_hallucination_passes_quality_only_eval(self):
        """
        FAILURE: A fluent, plausible-sounding hallucination passes an evaluator
        that only checks answer quality (length, coherence) but not grounding.
        """
        # Hallucinated response: fluent, confident, completely fabricated
        hallucinated_response = (
            "According to our updated policy from last month, customers now have "
            "60 days to return any item for a full refund. This was expanded from "
            "the previous 30-day window following customer feedback."
        )
        context = [get_doc("refund_policy")]
        query   = "What is the return window?"

        # Evaluator WITHOUT groundedness (only length/coherence — naive)
        naive_eval = {"passed": len(hallucinated_response) > 50 and "day" in hallucinated_response}

        # Evaluator WITH groundedness
        strict_evaluator = Evaluator(
            faithfulness_threshold=0.4,
            groundedness_threshold=0.7,
        )
        strict_eval = strict_evaluator.evaluate(query, hallucinated_response, context)

        print(f"\n[FAILURE DEMO] Hallucinated response:")
        print(f"  '{hallucinated_response}'")
        print(f"\n  Naive eval (length+keyword): passed={naive_eval['passed']}")
        print(f"  Strict eval (faithfulness+groundedness): passed={strict_eval.passed}")
        print(f"  Faithfulness: {strict_eval.faithfulness:.2f}")
        print(f"  Groundedness: {strict_eval.groundedness:.2f}")

        assert naive_eval["passed"] == True, "Naive evaluator should (wrongly) pass the hallucination"
        assert strict_eval.passed == False, "Strict evaluator should catch the hallucination"

    def test_groundedness_score_increases_with_context_support(self):
        """
        Groundedness score should track with how much the response is supported by context.
        """
        evaluator = Evaluator()
        context = [get_doc("refund_policy")]
        query   = "When are refunds processed?"

        responses = {
            "completely_made_up": "Refunds are processed instantly via blockchain technology.",
            "partially_grounded": "Refunds take about a week, based on our standard policy.",
            "fully_grounded":     "Approved refunds are processed within 5-7 business days to the original payment method.",
        }

        print(f"\n[FIX VERIFICATION] Groundedness tracking:")
        for label, response in responses.items():
            r = evaluator.evaluate(query, response, context)
            print(f"  {label:<30} groundedness={r.groundedness:.2f}, faithfulness={r.faithfulness:.2f}")
