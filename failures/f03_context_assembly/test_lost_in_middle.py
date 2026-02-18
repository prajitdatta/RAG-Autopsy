"""
failures/f03_context_assembly/test_lost_in_middle.py

FAILURE: LLMs pay more attention to the beginning and end of context.
         The correct chunk placed in the middle of a long context is ignored.

ROOT CAUSE:
    Empirically shown by Liu et al. (2023) "Lost in the Middle":
    LLM performance on multi-document QA degrades when the relevant document
    is placed in the middle of the context window vs at the start or end.

    Most RAG pipelines naively concatenate chunks in retrieval rank order
    (rank 1 first). If rank 1 is a false positive and the real answer is at
    rank 3 (middle), the LLM often ignores it.

IMPACT:
    Correct answer present in context → LLM ignores it → wrong answer.
    This is a generation/context-assembly failure, not a retrieval failure.

RUN:
    pytest failures/f03_context_assembly/test_lost_in_middle.py -v
"""

import pytest
from src.generator import MockGenerator
from src.evaluator import Evaluator
from data.sample_docs import get_doc
from data.sample_queries import get_query


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_context_with_answer_at_position(
    answer_chunk: str,
    noise_chunk: str,
    position: str,  # "beginning", "middle", "end"
    n_noise: int = 4,
) -> list[str]:
    """
    Build a context window with the answer chunk at a specific position,
    surrounded by noise chunks of similar length.
    """
    noise = [noise_chunk] * n_noise

    if position == "beginning":
        return [answer_chunk] + noise
    elif position == "end":
        return noise + [answer_chunk]
    elif position == "middle":
        mid = n_noise // 2
        return noise[:mid] + [answer_chunk] + noise[mid:]
    else:
        raise ValueError(f"Unknown position: {position}")


ANSWER_CHUNK = """
Section 3: Refund Processing
Approved refunds are processed within 5-7 business days.
Refunds are issued to the original payment method only.
Credit card refunds may take an additional 3-5 business days to appear on statements.
Cash payments are refunded via company check mailed within 10 business days.
"""

NOISE_CHUNK = """
Section 1: General Return Window
Customers may return most items within 30 days of the purchase date.
Items must be in their original condition, unused, and in original packaging.
Proof of purchase is required for all returns.
"""


class TestLostInMiddle:

    def test_answer_found_at_beginning(self):
        """
        BASELINE: When the answer chunk is first, the MockGenerator finds it.
        """
        generator = MockGenerator(mode="faithful")
        evaluator = Evaluator()
        query_data = get_query("q03")

        context = build_context_with_answer_at_position(
            ANSWER_CHUNK, NOISE_CHUNK, position="beginning"
        )

        result = generator.generate(query_data["query"], context)
        eval_result = evaluator.evaluate(
            query_data["query"], result.response, context,
            key_facts=query_data["key_facts"],
        )

        print(f"\n[BASELINE] Answer at BEGINNING — coverage: {eval_result.key_fact_coverage:.2f}")
        print(f"  Response: '{result.response[:100]}'")

    def test_answer_found_at_end(self):
        """
        END: LLMs also pay attention to the end of context.
        Usually similar to beginning.
        """
        generator = MockGenerator(mode="faithful")
        evaluator = Evaluator()
        query_data = get_query("q03")

        context = build_context_with_answer_at_position(
            ANSWER_CHUNK, NOISE_CHUNK, position="end"
        )

        result = generator.generate(query_data["query"], context)
        eval_result = evaluator.evaluate(
            query_data["query"], result.response, context,
            key_facts=query_data["key_facts"],
        )

        print(f"\n[END] Answer at END — coverage: {eval_result.key_fact_coverage:.2f}")
        print(f"  Response: '{result.response[:100]}'")

    def test_answer_lost_in_middle(self):
        """
        FAILURE: The answer chunk is in the middle, surrounded by noise.
        A real LLM often ignores middle-positioned answers. Our MockGenerator
        searches all chunks so demonstrates the principle; a real LLM
        integration test would show stronger degradation.
        """
        generator = MockGenerator(mode="faithful")
        evaluator = Evaluator()
        query_data = get_query("q03")

        context = build_context_with_answer_at_position(
            ANSWER_CHUNK, NOISE_CHUNK, position="middle", n_noise=6
        )

        print(f"\n[FAILURE DEMO] Context structure (7 chunks):")
        for i, c in enumerate(context):
            is_answer = "← ANSWER HERE" if ANSWER_CHUNK.strip()[:30] in c else ""
            print(f"  Position {i}: '{c.strip()[:50]}...' {is_answer}")

        result = generator.generate(query_data["query"], context)
        eval_result = evaluator.evaluate(
            query_data["query"], result.response, context,
            key_facts=query_data["key_facts"],
        )

        print(f"\n  Coverage with answer in middle: {eval_result.key_fact_coverage:.2f}")
        print(f"  Response: '{result.response[:100]}'")
        print(f"\n  NOTE: In a real LLM, middle-positioned answers score 20-40% lower.")
        print(f"  See: Liu et al. (2023) 'Lost in the Middle', arxiv.org/abs/2307.03172")

    def test_position_sweep_shows_degradation(self):
        """
        DEMONSTRATION: Sweep through all positions and show coverage at each.
        Real LLMs show U-shaped performance curve (high at start/end, low in middle).
        """
        generator = MockGenerator(mode="faithful")
        evaluator = Evaluator()
        query_data = get_query("q03")

        results = {}
        for n_noise in [2, 4, 6]:
            for position in ["beginning", "middle", "end"]:
                context = build_context_with_answer_at_position(
                    ANSWER_CHUNK, NOISE_CHUNK, position=position, n_noise=n_noise
                )
                result = generator.generate(query_data["query"], context)
                eval_result = evaluator.evaluate(
                    query_data["query"], result.response, context,
                    key_facts=query_data["key_facts"],
                )
                results[f"{position}_{n_noise}noise"] = eval_result.key_fact_coverage

        print(f"\n[FAILURE DEMO] Key fact coverage by answer position:")
        print(f"  {'Config':<25} | {'Coverage':>10}")
        print(f"  {'-'*38}")
        for config, cov in results.items():
            bar = "█" * int(cov * 20)
            print(f"  {config:<25} | {cov:>8.2f}  {bar}")


class TestContextAssemblyFixes:

    def test_fix_put_highest_ranked_chunk_first(self):
        """
        FIX #1: Always put the most relevant chunk at the beginning of context.
        Simple reordering — no extra infrastructure needed.
        """
        generator = MockGenerator(mode="faithful")
        evaluator = Evaluator()
        query_data = get_query("q03")

        # Simulate wrong ordering (answer at position 3 by retrieval rank)
        wrong_order = [NOISE_CHUNK, NOISE_CHUNK, ANSWER_CHUNK, NOISE_CHUNK]
        # Fix: move highest-relevance chunk to front (simulate reranker output)
        correct_order = [ANSWER_CHUNK, NOISE_CHUNK, NOISE_CHUNK, NOISE_CHUNK]

        for label, context in [("wrong order", wrong_order), ("answer first", correct_order)]:
            result = generator.generate(query_data["query"], context)
            eval_r = evaluator.evaluate(
                query_data["query"], result.response, context,
                key_facts=query_data["key_facts"],
            )
            print(f"\n  [{label}] coverage: {eval_r.key_fact_coverage:.2f}")

    def test_fix_use_smaller_context_window(self):
        """
        FIX #2: Use only the top 2-3 chunks instead of top-10.
        Less context = less noise = less chance of losing the answer.
        """
        generator = MockGenerator(mode="faithful")
        evaluator = Evaluator()
        query_data = get_query("q03")

        for context_size in [1, 2, 3, 5, 8]:
            # Build context with answer at position 1 (after noise)
            chunks = [NOISE_CHUNK] * (context_size - 1) + [ANSWER_CHUNK]
            used   = chunks[:context_size]
            result = generator.generate(query_data["query"], used)
            eval_r = evaluator.evaluate(
                query_data["query"], result.response, used,
                key_facts=query_data["key_facts"],
            )
            print(f"  context_size={context_size}: coverage={eval_r.key_fact_coverage:.2f}")
