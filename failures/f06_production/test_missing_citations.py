"""
failures/f06_production/test_missing_citations.py

FAILURE: RAG answers are not linked back to source documents.
         Users cannot verify claims. Hallucinations are indistinguishable
         from grounded answers. Trust collapses.

RUN:
    pytest failures/f06_production/test_missing_citations.py -v
"""

import pytest
from src.generator import MockGenerator
from src.evaluator import Evaluator
from data.sample_docs import get_doc
from data.sample_queries import QUERIES


class TestMissingCitations:

    def test_no_citations_makes_hallucinations_undetectable(self):
        """
        FAILURE: Without citations, the user cannot distinguish a grounded
        answer from a hallucinated one. Both look identical.
        """
        faithful_gen     = MockGenerator(mode="faithful")
        hallucinating_gen = MockGenerator(mode="hallucinating")

        context = [get_doc("refund_policy")]
        query   = "What is the return window?"

        faithful_result     = faithful_gen.generate(query, context)
        hallucinating_result = hallucinating_gen.generate(query, context)

        print(f"\n[FAILURE DEMO] Without citations — can you tell which is hallucinated?")
        print(f"\n  Response A: '{faithful_result.response}'")
        print(f"  Response B: '{hallucinating_result.response}'")
        print(f"\n  (Response B is hallucinated — but without citations, user can't tell)")

        # Without strict citation requirement, both "pass" a naive check
        naive_pass_a = len(faithful_result.response) > 20
        naive_pass_b = len(hallucinating_result.response) > 20
        assert naive_pass_a and naive_pass_b, (
            "Both responses pass naive (length-only) quality check — "
            "demonstrating that citations are needed to distinguish them."
        )

    def test_citation_requirement_forces_traceability(self):
        """
        FIX: With citation requirement enforced, hallucinating generators
        are caught and fail evaluation. Users can trust cited responses.
        """
        faithful_gen     = MockGenerator(mode="faithful")
        hallucinating_gen = MockGenerator(mode="hallucinating")
        evaluator = Evaluator(require_citations=True)

        context = [get_doc("refund_policy")]
        query   = "What is the return policy for defective items?"

        faithful_result     = faithful_gen.generate(query, context)
        hallucinating_result = hallucinating_gen.generate(query, context)

        faithful_eval     = evaluator.evaluate(query, faithful_result.response,     context, citations=faithful_result.citations)
        hallucinating_eval = evaluator.evaluate(query, hallucinating_result.response, context, citations=hallucinating_result.citations)

        print(f"\n[FIX VERIFICATION] Evaluation with citation requirement:")
        print(f"  Faithful (has citations):      passed={faithful_eval.passed}, "
              f"citations={faithful_result.citations}")
        print(f"  Hallucinating (no citations):  passed={hallucinating_eval.passed}, "
              f"citations={hallucinating_result.citations}")

        assert not hallucinating_eval.passed, (
            "Hallucinating generator should fail when citations are required."
        )

    def test_citation_coverage_across_all_queries(self):
        """
        Measure citation coverage across all test queries.
        In production, 100% of responses should have source citations.
        """
        generator = MockGenerator(mode="faithful")
        context_map = {q["source_doc"]: get_doc(q["source_doc"]) for q in QUERIES}

        cited    = 0
        uncited  = 0

        print(f"\n[FAILURE DEMO] Citation coverage across {len(QUERIES)} queries:")
        for q in QUERIES:
            context = [context_map[q["source_doc"]]]
            result  = generator.generate(q["query"], context)
            has_citation = len(result.citations) > 0
            if has_citation:
                cited   += 1
            else:
                uncited += 1

        citation_rate = cited / len(QUERIES)
        print(f"  Queries with citations: {cited}/{len(QUERIES)} ({citation_rate:.0%})")
        print(f"  Queries without:        {uncited}/{len(QUERIES)}")

        if citation_rate < 1.0:
            print(f"  ❌ {uncited} responses are not traceable to source")
        else:
            print(f"  ✅ All responses are traceable")
