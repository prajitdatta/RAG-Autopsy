"""
src/chunker.py

Text chunking with four strategies. This is the real implementation
that tests compare against — not a toy. Every test imports from here.

Strategies:
    fixed      — Split on character count with overlap. Fast but breaks sentences.
    sentence   — Split on sentence boundaries. Respects language units.
    recursive  — Try paragraphs → sentences → words. Best general-purpose.
    semantic   — Detect topic shifts via cosine similarity. Highest quality, slowest.
"""

import re
import math
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    index: int
    start_char: int
    end_char: int
    strategy: str
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Rough token count: 1 token ≈ 4 chars."""
        return max(1, len(self.text) // 4)

    def __repr__(self):
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(index={self.index}, tokens≈{self.token_estimate}, text='{preview}...')"


class Chunker:
    """
    Unified chunking interface. Pass strategy= to switch approaches.

    Args:
        strategy:             "fixed" | "sentence" | "recursive" | "semantic"
        chunk_size:           Target chunk size in characters (fixed/recursive)
        overlap:              Overlap between chunks in characters (fixed)
        max_sentences:        Max sentences per chunk (sentence strategy)
        similarity_threshold: Topic-shift threshold for semantic strategy (0–1)
    """

    STRATEGIES = ("fixed", "sentence", "recursive", "semantic")

    def __init__(
        self,
        strategy: str = "recursive",
        chunk_size: int = 512,
        overlap: int = 64,
        max_sentences: int = 5,
        similarity_threshold: float = 0.85,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}, got '{strategy}'")
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_sentences = max_sentences
        self.similarity_threshold = similarity_threshold

    def chunk(self, text: str) -> list[Chunk]:
        """Split text into chunks using the configured strategy."""
        text = text.strip()
        if not text:
            return []

        if self.strategy == "fixed":
            raw = self._fixed(text)
        elif self.strategy == "sentence":
            raw = self._sentence(text)
        elif self.strategy == "recursive":
            raw = self._recursive(text, self.chunk_size)
        elif self.strategy == "semantic":
            raw = self._semantic(text)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return [
            Chunk(
                text=t,
                index=i,
                start_char=text.find(t),
                end_char=text.find(t) + len(t),
                strategy=self.strategy,
            )
            for i, t in enumerate(raw)
            if t.strip()
        ]

    # ── Fixed ─────────────────────────────────────────────────────────────────

    def _fixed(self, text: str) -> list[str]:
        """
        Split on fixed character count with overlap.
        PROBLEM: Cuts mid-sentence, mid-word. Fast but semantically broken.
        Used in f01_chunking tests to demonstrate the failure.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap
        return chunks

    # ── Sentence-aware ────────────────────────────────────────────────────────

    def _sentence(self, text: str) -> list[str]:
        """
        Split on sentence boundaries. Groups max_sentences sentences per chunk.
        Better than fixed — respects language units.
        """
        sentences = self._split_sentences(text)
        chunks = []
        buffer = []
        for sent in sentences:
            buffer.append(sent)
            if len(buffer) >= self.max_sentences:
                chunks.append(" ".join(buffer))
                # Keep last sentence as overlap
                buffer = buffer[-1:]
        if buffer:
            chunks.append(" ".join(buffer))
        return chunks

    # ── Recursive ─────────────────────────────────────────────────────────────

    def _recursive(self, text: str, max_size: int) -> list[str]:
        """
        Hierarchical splitting: try paragraph → sentence → word.
        Best general-purpose strategy.
        """
        if len(text) <= max_size:
            return [text]

        # Try splitting on double newline (paragraphs)
        parts = re.split(r"\n\n+", text)
        if len(parts) > 1:
            return self._merge_splits(parts, max_size, separator="\n\n")

        # Try splitting on single newline
        parts = text.split("\n")
        if len(parts) > 1:
            return self._merge_splits(parts, max_size, separator="\n")

        # Try splitting on sentences
        parts = self._split_sentences(text)
        if len(parts) > 1:
            return self._merge_splits(parts, max_size, separator=" ")

        # Last resort: split on words
        parts = text.split(" ")
        return self._merge_splits(parts, max_size, separator=" ")

    def _merge_splits(self, parts: list[str], max_size: int, separator: str) -> list[str]:
        """Merge small splits back together up to max_size."""
        chunks = []
        current = ""
        for part in parts:
            candidate = (current + separator + part).strip() if current else part
            if len(candidate) <= max_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If single part is too big, recurse
                if len(part) > max_size:
                    sub = self._recursive(part, max_size)
                    chunks.extend(sub[:-1])
                    current = sub[-1] if sub else ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks

    # ── Semantic ──────────────────────────────────────────────────────────────

    def _semantic(self, text: str) -> list[str]:
        """
        Split on detected topic shifts using cosine similarity between
        adjacent sentence embeddings. Groups semantically coherent sentences.
        Uses TF-IDF-style vectors (no external embedding API needed).
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return sentences

        vectors = [self._tfidf_vector(s) for s in sentences]
        chunks = []
        buffer = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = self._cosine(vectors[i - 1], vectors[i])
            if sim >= self.similarity_threshold:
                buffer.append(sentences[i])
            else:
                # Topic shift detected
                chunks.append(" ".join(buffer))
                buffer = [sentences[i]]

        if buffer:
            chunks.append(" ".join(buffer))
        return chunks

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using regex (no NLTK dependency)."""
        pattern = r"(?<=[.!?])\s+(?=[A-Z0-9\-])"
        sentences = re.split(pattern, text)
        # Also split on newlines that look like new logical units
        result = []
        for s in sentences:
            parts = s.split("\n")
            result.extend(p.strip() for p in parts if p.strip())
        return result

    @staticmethod
    def _tfidf_vector(text: str) -> dict[str, float]:
        """Compute a simple TF-IDF-style word vector."""
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        stopwords = {
            "the", "and", "for", "that", "this", "with", "are", "was",
            "not", "have", "has", "had", "its", "may", "per", "any",
        }
        tf: dict[str, float] = {}
        for w in words:
            if w not in stopwords:
                tf[w] = tf.get(w, 0) + 1
        total = sum(tf.values()) or 1
        return {w: c / total for w, c in tf.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        """Cosine similarity between two sparse vectors."""
        shared = set(a) & set(b)
        dot = sum(a[w] * b[w] for w in shared)
        norm_a = math.sqrt(sum(v * v for v in a.values())) or 1e-9
        norm_b = math.sqrt(sum(v * v for v in b.values())) or 1e-9
        return dot / (norm_a * norm_b)
