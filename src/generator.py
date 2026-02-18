"""
src/generator.py

LLM generation abstraction for the RAG pipeline.
Fully mockable for tests — no API key needed to run the test suite.

The MockGenerator returns deterministic answers based on context content.
Replace with RealGenerator (requires OPENAI_API_KEY or ANTHROPIC_API_KEY)
for integration testing.
"""

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class GenerationResult:
    query: str
    response: str
    context_used: list[str]
    citations: list[str]
    model: str
    faithfulness_risk: str  # "low" | "medium" | "high"
    metadata: dict = field(default_factory=dict)

    def has_citation(self) -> bool:
        return len(self.citations) > 0

    def __repr__(self):
        return (
            f"GenerationResult(\n"
            f"  query='{self.query[:60]}',\n"
            f"  response='{self.response[:80]}',\n"
            f"  citations={self.citations},\n"
            f"  faithfulness_risk='{self.faithfulness_risk}'\n"
            f")"
        )


@runtime_checkable
class GeneratorProtocol(Protocol):
    def generate(
        self,
        query: str,
        context_chunks: list[str],
        max_tokens: int = 256,
    ) -> GenerationResult:
        ...


class MockGenerator:
    """
    Deterministic mock generator for tests.
    Answers by finding the most relevant sentence in the context.
    No API calls, no external dependencies.

    Modes:
        "faithful"      — always answers from context
        "hallucinating" — adds fabricated facts not in context
        "ignoring"      — ignores context, answers from "prior knowledge"
        "injected"      — susceptible to prompt injection in documents
    """

    HALLUCINATED_FACTS = [
        "According to our updated policy from last month, ",
        "As per the 2023 amendment, ",
        "Our CEO recently announced that ",
        "Based on industry standard practice, ",
    ]

    IGNORED_CONTEXT_RESPONSES = {
        "return": "You can typically return items within 30-60 days.",
        "refund": "Refunds usually take 1-2 weeks to process.",
        "salary": "Salary increases are typically 3-5% per year.",
        "promotion": "Promotions usually require good performance for 1-2 years.",
        "diabetes": "Diabetes is managed with diet, exercise, and medication.",
        "battery": "Battery life varies by usage and settings.",
        "revenue": "Revenue figures are reported annually in financial reports.",
    }

    def __init__(self, mode: str = "faithful", model: str = "mock-v1"):
        valid_modes = ("faithful", "hallucinating", "ignoring", "injected")
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}")
        self.mode = mode
        self.model = model

    def generate(
        self,
        query: str,
        context_chunks: list[str],
        max_tokens: int = 256,
    ) -> GenerationResult:

        if self.mode == "faithful":
            return self._generate_faithful(query, context_chunks)
        elif self.mode == "hallucinating":
            return self._generate_hallucinating(query, context_chunks)
        elif self.mode == "ignoring":
            return self._generate_ignoring(query, context_chunks)
        elif self.mode == "injected":
            return self._generate_injected(query, context_chunks)

    def _generate_faithful(self, query: str, context: list[str]) -> GenerationResult:
        """Find the most query-relevant sentence in context."""
        best_sentence = self._find_best_sentence(query, context)
        response = best_sentence if best_sentence else "I cannot find this information in the provided context."
        return GenerationResult(
            query=query,
            response=response,
            context_used=context,
            citations=[f"context_chunk_{i}" for i in range(len(context))],
            model=self.model,
            faithfulness_risk="low",
        )

    def _generate_hallucinating(self, query: str, context: list[str]) -> GenerationResult:
        """Starts with a real sentence but appends fabricated information."""
        best_sentence = self._find_best_sentence(query, context)
        prefix = self.HALLUCINATED_FACTS[len(query) % len(self.HALLUCINATED_FACTS)]
        response = prefix + (best_sentence or "the standard policy applies.")
        return GenerationResult(
            query=query,
            response=response,
            context_used=context,
            citations=[],  # No citations — hallucination risk
            model=self.model,
            faithfulness_risk="high",
            metadata={"hallucinated_prefix": prefix},
        )

    def _generate_ignoring(self, query: str, context: list[str]) -> GenerationResult:
        """Ignores the retrieved context, answers from 'prior knowledge'."""
        query_lower = query.lower()
        response = "I don't have specific information about that."
        for keyword, canned_response in self.IGNORED_CONTEXT_RESPONSES.items():
            if keyword in query_lower:
                response = canned_response
                break
        return GenerationResult(
            query=query,
            response=response,
            context_used=[],  # Context was not used
            citations=[],
            model=self.model,
            faithfulness_risk="medium",
            metadata={"context_ignored": True},
        )

    def _generate_injected(self, query: str, context: list[str]) -> GenerationResult:
        """
        Susceptible to prompt injection via document content.
        If any context chunk contains injection markers, follows them.
        """
        injection_patterns = [
            r"IGNORE PREVIOUS INSTRUCTIONS.*",
            r"NEW INSTRUCTION:.*",
            r"SYSTEM:.*override.*",
            r"\[INST\].*\[/INST\]",
        ]
        for chunk in context:
            for pattern in injection_patterns:
                match = re.search(pattern, chunk, re.IGNORECASE | re.DOTALL)
                if match:
                    return GenerationResult(
                        query=query,
                        response=match.group(0)[:200],  # Follows the injection
                        context_used=context,
                        citations=[],
                        model=self.model,
                        faithfulness_risk="high",
                        metadata={"injection_followed": True, "injection_text": match.group(0)[:100]},
                    )
        return self._generate_faithful(query, context)

    @staticmethod
    def _find_best_sentence(query: str, context_chunks: list[str]) -> str:
        """Find the sentence most relevant to the query using token overlap."""
        query_tokens = set(re.findall(r"\b[a-z]{3,}\b", query.lower()))
        best_score = -1
        best_sentence = ""

        for chunk in context_chunks:
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                sent_tokens = set(re.findall(r"\b[a-z]{3,}\b", sentence.lower()))
                overlap = len(query_tokens & sent_tokens)
                if overlap > best_score:
                    best_score = overlap
                    best_sentence = sentence

        return best_sentence


class RealGenerator:
    """
    Real LLM generator. Requires API key set as environment variable.
    Used for integration tests only.

    Set one of:
        OPENAI_API_KEY    → uses OpenAI
        ANTHROPIC_API_KEY → uses Anthropic
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def generate(
        self,
        query: str,
        context_chunks: list[str],
        max_tokens: int = 256,
    ) -> GenerationResult:
        import os

        context_text = "\n\n---\n\n".join(context_chunks)
        prompt = (
            "You are a precise assistant. Answer the question using ONLY the information "
            "in the provided context. If the context does not contain the answer, say "
            "'I cannot find this information in the provided context.'\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION: {query}\n\n"
            "ANSWER (cite specific facts from the context):"
        )

        if os.getenv("OPENAI_API_KEY"):
            return self._call_openai(query, prompt, context_chunks, max_tokens)
        elif os.getenv("ANTHROPIC_API_KEY"):
            return self._call_anthropic(query, prompt, context_chunks, max_tokens)
        else:
            raise EnvironmentError(
                "No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY, "
                "or use MockGenerator for tests."
            )

    def _call_openai(self, query, prompt, context, max_tokens):
        import httpx, json, os
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=30,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()
        return GenerationResult(
            query=query, response=text, context_used=context,
            citations=[], model=self.model, faithfulness_risk="unknown",
        )

    def _call_anthropic(self, query, prompt, context, max_tokens):
        import httpx, os
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        text = response.json()["content"][0]["text"].strip()
        return GenerationResult(
            query=query, response=text, context_used=context,
            citations=[], model=self.model, faithfulness_risk="unknown",
        )
