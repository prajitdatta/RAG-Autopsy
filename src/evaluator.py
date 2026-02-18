"""
src/evaluator.py

RAG evaluation metrics.

Metrics:
    faithfulness      — Is the answer supported by the retrieved context?
    relevance         — Is the retrieved context relevant to the query?
    groundedness      — Does the answer contain only facts from the context?
    key_fact_coverage — Are key facts from ground truth present in the answer?
    citation_quality  — Are claims backed by citations?
"""

import re
import math
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    query: str
    faithfulness: float         # 0-1: answer supported by context
    relevance: float            # 0-1: context relevant to query
    groundedness: float         # 0-1: answer grounded (no hallucination indicators)
    key_fact_coverage: float    # 0-1: fraction of key facts present
    has_citations: bool
    passed: bool
    reasons: list[str]

    def summary(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"{status} | faithfulness={self.faithfulness:.2f} | "
            f"relevance={self.relevance:.2f} | "
            f"groundedness={self.groundedness:.2f} | "
            f"key_facts={self.key_fact_coverage:.2f}"
        )


class Evaluator:
    """
    Evaluate RAG pipeline outputs without an LLM-as-judge.
    Uses heuristic methods — fast, deterministic, suitable for CI.

    For production: replace with an LLM-as-judge for higher accuracy.

    Args:
        faithfulness_threshold:   Minimum faithfulness to pass (default 0.3)
        relevance_threshold:      Minimum relevance to pass (default 0.3)
        groundedness_threshold:   Minimum groundedness to pass (default 0.5)
        require_citations:        Whether citations are required (default False)
    """

    # Phrases that strongly suggest hallucination
    HALLUCINATION_SIGNALS = [
        "according to our updated policy",
        "as per the 2023 amendment",
        "recently announced",
        "our ceo",
        "industry standard practice",
        "typically",
        "usually",
        "generally speaking",
        "in most cases",
        "it is well known",
    ]

    def __init__(
        self,
        faithfulness_threshold: float = 0.3,
        relevance_threshold: float = 0.3,
        groundedness_threshold: float = 0.5,
        require_citations: bool = False,
    ):
        self.faithfulness_threshold = faithfulness_threshold
        self.relevance_threshold = relevance_threshold
        self.groundedness_threshold = groundedness_threshold
        self.require_citations = require_citations

    def evaluate(
        self,
        query: str,
        response: str,
        context_chunks: list[str],
        key_facts: list[str] | None = None,
        citations: list[str] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a RAG response.

        Args:
            query:          The user's question
            response:       The generated answer
            context_chunks: The retrieved context chunks
            key_facts:      Expected facts that should be in the response
            citations:      Citations the generator included

        Returns:
            EvaluationResult with per-metric scores and pass/fail
        """
        faithfulness = self._faithfulness(response, context_chunks)
        relevance    = self._relevance(query, context_chunks)
        groundedness = self._groundedness(response)
        key_fact_cov = self._key_fact_coverage(response, key_facts or [])
        has_citations = bool(citations)

        reasons = []
        passed = True

        if faithfulness < self.faithfulness_threshold:
            reasons.append(
                f"Low faithfulness ({faithfulness:.2f} < {self.faithfulness_threshold}): "
                f"response may not be supported by context"
            )
            passed = False
        if relevance < self.relevance_threshold:
            reasons.append(
                f"Low relevance ({relevance:.2f} < {self.relevance_threshold}): "
                f"retrieved context may not address the query"
            )
            passed = False
        if groundedness < self.groundedness_threshold:
            reasons.append(
                f"Low groundedness ({groundedness:.2f} < {self.groundedness_threshold}): "
                f"response contains hallucination signals"
            )
            passed = False
        if self.require_citations and not has_citations:
            reasons.append("No citations provided — required by evaluation policy")
            passed = False
        if key_facts and key_fact_cov < 0.5:
            reasons.append(
                f"Poor key fact coverage ({key_fact_cov:.2f}): "
                f"answer is missing important facts"
            )
            passed = False

        return EvaluationResult(
            query=query,
            faithfulness=faithfulness,
            relevance=relevance,
            groundedness=groundedness,
            key_fact_coverage=key_fact_cov,
            has_citations=has_citations,
            passed=passed,
            reasons=reasons,
        )

    def _faithfulness(self, response: str, context_chunks: list[str]) -> float:
        """
        Measure token overlap between response and context.
        Low overlap = response contains content not in context = potential hallucination.
        """
        if not context_chunks:
            return 0.0
        response_tokens = self._tokenize(response)
        context_text = " ".join(context_chunks)
        context_tokens = set(self._tokenize(context_text))

        if not response_tokens:
            return 0.0

        supported = sum(1 for t in response_tokens if t in context_tokens)
        return supported / len(response_tokens)

    def _relevance(self, query: str, context_chunks: list[str]) -> float:
        """Measure how well the context matches the query."""
        if not context_chunks:
            return 0.0
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return 0.0

        scores = []
        for chunk in context_chunks:
            chunk_tokens = set(self._tokenize(chunk))
            if not chunk_tokens:
                continue
            intersection = query_tokens & chunk_tokens
            union = query_tokens | chunk_tokens
            jaccard = len(intersection) / len(union) if union else 0
            scores.append(jaccard)

        return max(scores) if scores else 0.0

    def _groundedness(self, response: str) -> float:
        """
        Detect hallucination signals in the response.
        Returns 1.0 if no signals found, lower values for each signal detected.
        """
        response_lower = response.lower()
        signals_found = sum(
            1 for signal in self.HALLUCINATION_SIGNALS
            if signal in response_lower
        )
        return max(0.0, 1.0 - (signals_found * 0.3))

    def _key_fact_coverage(self, response: str, key_facts: list[str]) -> float:
        """Fraction of key facts present in the response."""
        if not key_facts:
            return 1.0
        response_lower = response.lower()
        found = sum(1 for fact in key_facts if fact.lower() in response_lower)
        return found / len(key_facts)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        stopwords = {
            "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
            "of", "and", "or", "but", "with", "by", "from", "be", "are", "i",
            "not", "if", "this", "that", "was", "has", "had", "may", "can",
        }
        tokens = re.findall(r"\b[a-z0-9]{2,}\b", text.lower())
        return [t for t in tokens if t not in stopwords]
