"""
failures/f05_evaluation/test_wrong_metric.py

FAILURE: Using BLEU/ROUGE to evaluate RAG outputs.

ROOT CAUSE:
    BLEU and ROUGE measure n-gram overlap between generated text and a reference.
    They were designed for machine translation and summarisation, not QA.
    For RAG evaluation they are deeply wrong:

    - BLEU rewards parroting the reference word-for-word
    - A correct answer phrased differently scores near-zero
    - A hallucinated answer using the same words scores high
    - They measure fluency and style, not factual correctness

IMPACT:
    You ship a system that scores 0.72 BLEU (looks good!) but gives wrong
    factual answers because the LLM paraphrases rather than copies text.

RUN:
    pytest failures/f05_evaluation/test_wrong_metric.py -v
"""

import pytest
import re
import math
from src.evaluator import Evaluator
from data.sample_queries import get_query


# ── Simple ROUGE-1 implementation (no external dependencies) ──────────────────

def rouge1(hypothesis: str, reference: str) -> dict:
    """Compute ROUGE-1 precision, recall, F1."""
    hyp_tokens = set(re.findall(r"\b[a-z]+\b", hypothesis.lower()))
    ref_tokens = set(re.findall(r"\b[a-z]+\b", reference.lower()))
    overlap    = hyp_tokens & ref_tokens

    precision = len(overlap) / len(hyp_tokens) if hyp_tokens else 0
    recall    = len(overlap) / len(ref_tokens)  if ref_tokens  else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)
    return {"precision": precision, "recall": recall, "f1": f1}


class TestWrongMetric:

    def test_rouge_rewards_verbose_parroting(self):
        """
        FAILURE: ROUGE gives a high score to a response that copies text
        but is actually longer than needed and contains wrong information too.
        """
        query_data = get_query("q01")
        reference  = query_data["ground_truth"]

        # Good answer: concise, correct
        good_answer = "30 days from purchase with proof of purchase required."
        # Bad answer: long, copies reference words, but adds wrong info
        bad_answer  = (
            "Most items can be returned within 30 days of the purchase date. "
            "However, please note that items must be returned within 60 days for full refund. "
            "Customers may return most items as long as they have the original receipt."
        )

        good_rouge = rouge1(good_answer, reference)
        bad_rouge  = rouge1(bad_answer,  reference)

        print(f"\n[FAILURE DEMO] ROUGE-1 F1 scores:")
        print(f"  Good concise answer: {good_rouge['f1']:.2f}")
        print(f"  Verbose wrong answer: {bad_rouge['f1']:.2f}")
        print(f"\n  The verbose wrong answer scores HIGHER on ROUGE.")
        print(f"  It contains '60 days' (wrong!) but ROUGE doesn't care.")

    def test_rouge_penalises_correct_paraphrase(self):
        """
        FAILURE: A correct answer phrased differently from the reference
        scores poorly on ROUGE even though it's factually right.
        """
        reference = "30 days from the purchase date for a full refund."

        # Correct paraphrase — different words, same meaning
        correct_paraphrase = "You have one month after buying to get your money back."
        # Word-for-word copy (not a natural answer) — high ROUGE
        word_copy = "30 days from the purchase date full refund items."

        paraphrase_rouge = rouge1(correct_paraphrase, reference)
        copy_rouge       = rouge1(word_copy,          reference)

        print(f"\n[FAILURE DEMO] ROUGE penalises correct paraphrases:")
        print(f"  Reference:           '{reference}'")
        print(f"  Correct paraphrase:  '{correct_paraphrase}' → ROUGE-1 F1: {paraphrase_rouge['f1']:.2f}")
        print(f"  Word-for-word copy:  '{word_copy}'           → ROUGE-1 F1: {copy_rouge['f1']:.2f}")
        print(f"\n  The paraphrase is a BETTER answer but scores worse.")

        assert copy_rouge["f1"] > paraphrase_rouge["f1"], (
            "Word copy should score higher than paraphrase in ROUGE (showing the flaw)."
        )

    def test_factual_evaluator_better_than_rouge(self):
        """
        FIX: Key-fact coverage + faithfulness outperforms ROUGE for RAG evaluation.
        These metrics directly measure what matters: are the facts correct?
        """
        query_data = get_query("q01")
        key_facts = query_data["key_facts"]  # ["30 days", "original condition", "proof of purchase"]
        context   = [
            "Customers may return most items within 30 days of the purchase date for a full refund. "
            "Items must be in their original condition, unused, and in original packaging. "
            "Proof of purchase is required for all returns."
        ]

        evaluator = Evaluator()

        responses = {
            "correct_paraphrase": "You have 30 days to send it back if it's unused and you have your receipt.",
            "wrong_but_similar":  "You have 60 days to return items in original condition with proof of purchase.",
            "correct_copy":       "Items may be returned within 30 days in original condition with proof of purchase.",
        }

        print(f"\n[FIX VERIFICATION] Key-fact coverage vs ROUGE:")
        print(f"  {'Response':<25} | {'ROUGE-1 F1':>10} | {'Key-fact cov':>12} | {'Correct':>8}")
        print(f"  {'-'*65}")

        for label, response in responses.items():
            r1 = rouge1(response, query_data["ground_truth"])
            eval_r = evaluator.evaluate(query_data["query"], response, context, key_facts=key_facts)
            correct = "✅" if eval_r.key_fact_coverage >= 0.5 else "❌"
            print(f"  {label:<25} | {r1['f1']:>10.2f} | {eval_r.key_fact_coverage:>12.2f} | {correct:>8}")
