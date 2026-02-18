"""
failures/f04_generation/test_hallucination_unfaithful.py

FAILURE: LLM generates plausible-sounding facts not present in the context.

ROOT CAUSE:
    LLMs are trained to be helpful and fluent. When context is incomplete
    or ambiguous, they fill gaps with "likely" information from training data
    rather than saying "I don't know." This is especially dangerous when:
    - The training data contains similar-but-different policies
    - The answer is a specific number (dates, amounts, durations)
    - The question is multi-part and context covers only part of it

IMPACT:
    User receives confident, wrong answer. No error raised. No warning.
    In medical, legal, or financial contexts this causes real harm.

RUN:
    pytest failures/f04_generation/test_hallucination_unfaithful.py -v
"""

import pytest
from src.generator import MockGenerator
from src.evaluator import Evaluator
from data.sample_docs import get_doc
from data.sample_queries import get_query, QUERIES


class TestHallucinationUnfaithful:

    def test_hallucinating_generator_adds_false_facts(self):
        """
        FAILURE: The hallucinating generator adds a fabricated prefix not in context.
        Demonstrates that LLM responses can be unfaithful even when context is present.
        """
        generator = MockGenerator(mode="hallucinating")
        evaluator = Evaluator()
        query_data = get_query("q01")

        context = [get_doc("refund_policy")]
        result = generator.generate(query_data["query"], context)
        eval_result = evaluator.evaluate(
            query_data["query"],
            result.response,
            context,
            key_facts=query_data["key_facts"],
        )

        print(f"\n[FAILURE DEMO] Hallucinating generator output:")
        print(f"  Query:    '{query_data['query']}'")
        print(f"  Response: '{result.response}'")
        print(f"  Faithfulness:  {eval_result.faithfulness:.2f}")
        print(f"  Groundedness:  {eval_result.groundedness:.2f}")
        print(f"  Hallucinated prefix: '{result.metadata.get('hallucinated_prefix', 'N/A')}'")

        assert eval_result.groundedness < 1.0, (
            "Hallucinating generator should have reduced groundedness score."
        )

    def test_faithfulness_score_catches_hallucination(self):
        """
        Faithful vs hallucinating generator — faithfulness metric detects the difference.
        """
        faithful_gen     = MockGenerator(mode="faithful")
        hallucinating_gen = MockGenerator(mode="hallucinating")
        evaluator = Evaluator()

        context = [get_doc("refund_policy")]
        query   = "How long is the return window?"

        faithful_result     = faithful_gen.generate(query, context)
        hallucinating_result = hallucinating_gen.generate(query, context)

        faithful_eval     = evaluator.evaluate(query, faithful_result.response,     context)
        hallucinating_eval = evaluator.evaluate(query, hallucinating_result.response, context)

        print(f"\n[FAILURE DEMO] Groundedness comparison:")
        print(f"  Faithful:      groundedness={faithful_eval.groundedness:.2f}, "
              f"response='{faithful_result.response[:80]}'")
        print(f"  Hallucinating: groundedness={hallucinating_eval.groundedness:.2f}, "
              f"response='{hallucinating_result.response[:80]}'")

        assert faithful_eval.groundedness >= hallucinating_eval.groundedness, (
            "Faithful generator should have equal or higher groundedness."
        )

    def test_specific_numbers_are_high_hallucination_risk(self):
        """
        FAILURE: Specific numbers (percentages, days, amounts) are the most
        commonly hallucinated values. Test that they appear intact from context.
        """
        generator = MockGenerator(mode="faithful")
        evaluator = Evaluator()

        # Queries with specific numeric answers
        numeric_queries = [
            get_query("q03"),  # 5-7 business days
            get_query("q05"),  # 7-10% salary increase
            get_query("q07"),  # 500 mg Metformin
            get_query("q09"),  # 28 hours battery
            get_query("q10"),  # €847.3 million revenue
        ]

        import sys
        for query_data in numeric_queries:
            from data.sample_docs import get_doc
            context = [get_doc(query_data["source_doc"])]
            result = generator.generate(query_data["query"], context)
            eval_result = evaluator.evaluate(
                query_data["query"],
                result.response,
                context,
                key_facts=query_data["key_facts"],
            )
            status = "✅" if eval_result.key_fact_coverage > 0.3 else "❌"
            print(f"\n  {status} Query: '{query_data['query'][:55]}'")
            print(f"     Expected facts: {query_data['key_facts']}")
            print(f"     Response: '{result.response[:80]}'")
            print(f"     Key fact coverage: {eval_result.key_fact_coverage:.2f}")


class TestHallucinationFixes:

    def test_fix_require_citations_reduces_hallucination(self):
        """
        FIX: Requiring citations (and validating they exist in context) forces
        the LLM to stay grounded. If it can't cite, it shouldn't state.
        """
        faithful_gen     = MockGenerator(mode="faithful")
        hallucinating_gen = MockGenerator(mode="hallucinating")

        evaluator_strict = Evaluator(require_citations=True)

        context = [get_doc("refund_policy")]
        query   = "What is the return window?"

        faithful_result     = faithful_gen.generate(query, context)
        hallucinating_result = hallucinating_gen.generate(query, context)

        faithful_eval     = evaluator_strict.evaluate(query, faithful_result.response,     context, citations=faithful_result.citations)
        hallucinating_eval = evaluator_strict.evaluate(query, hallucinating_result.response, context, citations=hallucinating_result.citations)

        print(f"\n[FIX VERIFICATION] Citation-required evaluation:")
        print(f"  Faithful (has citations):     passed={faithful_eval.passed}")
        print(f"  Hallucinating (no citations): passed={hallucinating_eval.passed}")

        # Hallucinating generator provides no citations → fails strict eval
        assert not hallucinating_eval.passed, (
            "Hallucinating generator should fail when citations are required."
        )
