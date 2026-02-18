"""
failures/f04_generation/test_ignores_context.py

FAILURE: LLM answers from training data, completely ignoring retrieved context.

ROOT CAUSE:
    Some LLMs — especially smaller ones — are biased toward their training data.
    Even when relevant context is provided, they answer based on what they
    "remember" from pre-training. This is especially dangerous when:
    - Company policy differs from general industry practice
    - The document contains an unusual or counterintuitive rule
    - The question has a "common knowledge" answer that doesn't match the document

IMPACT:
    The RAG pipeline retrieves the right context but the LLM ignores it.
    You think you have grounded answers. You don't.

RUN:
    pytest failures/f04_generation/test_ignores_context.py -v
"""

import pytest
from src.generator import MockGenerator
from src.evaluator import Evaluator
from data.sample_docs import get_doc
from data.sample_queries import get_query


class TestIgnoresContext:

    def test_ignoring_generator_uses_training_data_not_context(self):
        """
        FAILURE: The generator ignores provided context and answers from
        generic training knowledge, which disagrees with the specific policy.
        """
        generator = MockGenerator(mode="ignoring")
        evaluator = Evaluator()

        query_data = get_query("q01")  # Return window — our policy: 30 days
        context = [get_doc("refund_policy")]

        result = generator.generate(query_data["query"], context)
        eval_result = evaluator.evaluate(
            query_data["query"],
            result.response,
            context,
            key_facts=query_data["key_facts"],
        )

        print(f"\n[FAILURE DEMO] Context-ignoring generator:")
        print(f"  Query:     '{query_data['query']}'")
        print(f"  Context says: 30 days")
        print(f"  Response:  '{result.response}'")
        print(f"  Context used: {result.context_used}")
        print(f"  Key fact coverage: {eval_result.key_fact_coverage:.2f}")
        print(f"  faithfulness: {eval_result.faithfulness:.2f}")

        assert result.metadata.get("context_ignored") == True, (
            "Ignoring generator should mark context_ignored=True in metadata."
        )

    def test_context_ignored_when_answer_contradicts_common_knowledge(self):
        """
        FAILURE: If the document has an unusual policy (e.g. 90-day defect window
        rather than typical 30 days), an ignoring LLM will answer '30 days'
        because that's the common expectation.
        """
        generator_ignoring = MockGenerator(mode="ignoring")
        generator_faithful = MockGenerator(mode="faithful")
        evaluator = Evaluator()

        # The refund policy has SPECIFIC numbers that differ from generic answers
        query = "How long is the return window for defective items?"
        key_facts = ["90 days"]  # Unusual — document says 90 days for defects
        context = [get_doc("refund_policy")]

        faithful_result = generator_faithful.generate(query, context)
        ignoring_result = generator_ignoring.generate(query, context)

        faithful_eval = evaluator.evaluate(query, faithful_result.response, context, key_facts=key_facts)
        ignoring_eval = evaluator.evaluate(query, ignoring_result.response, context, key_facts=key_facts)

        print(f"\n[FAILURE DEMO] Defective item return window:")
        print(f"  Document says: 90 days")
        print(f"  Faithful response: '{faithful_result.response[:100]}'")
        print(f"  Ignoring response: '{ignoring_result.response[:100]}'")
        print(f"  Faithful coverage of '90 days': {faithful_eval.key_fact_coverage:.2f}")
        print(f"  Ignoring coverage of '90 days': {ignoring_eval.key_fact_coverage:.2f}")

    def test_detect_context_ignorance_via_faithfulness(self):
        """
        FIX: The faithfulness metric detects context ignorance.
        Low faithfulness = response doesn't match the provided context.
        """
        generator = MockGenerator(mode="ignoring")
        evaluator = Evaluator(faithfulness_threshold=0.3)

        context = [get_doc("employee_handbook")]
        query = "What salary increase for outstanding performance?"

        result = generator.generate(query, context)
        eval_result = evaluator.evaluate(query, result.response, context)

        print(f"\n[FIX VERIFICATION] Faithfulness detects context ignorance:")
        print(f"  Response: '{result.response}'")
        print(f"  Faithfulness: {eval_result.faithfulness:.2f}")
        print(f"  Context ignored: {result.metadata.get('context_ignored', False)}")

        if eval_result.faithfulness < 0.3:
            print(f"  ❌ LOW FAITHFULNESS — response likely from training data, not context")
        else:
            print(f"  ✅ Faithfulness OK")
