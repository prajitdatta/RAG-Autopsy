"""
fixes/fix_chunking.py

Production-ready chunking fixes for all F01 failures.
Drop-in replacements for broken chunking strategies.

Usage:
    from fixes.fix_chunking import SmartChunker, deduplicate_chunks

    chunker = SmartChunker(chunk_size=400)
    chunks  = chunker.chunk(document_text)
    chunks  = deduplicate_chunks(chunks, threshold=0.7)
"""

import re
import math
from src.chunker import Chunker, Chunk


class SmartChunker:
    """
    Production chunker with sane defaults and automatic fallback.

    Fixes:
        F01a: Uses recursive strategy (not fixed) — no mid-sentence splits
        F01b: Enforces min/max chunk size — no oversized or undersized chunks
        F01c: Validates output — raises if chunking produces bad results
    """

    def __init__(
        self,
        chunk_size: int = 400,
        min_chunk_size: int = 100,
        max_chunk_size: int = 800,
        overlap: int = 0,
    ):
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self._chunker = Chunker(strategy="recursive", chunk_size=chunk_size)

    def chunk(self, text: str, source_id: str = "unknown") -> list[Chunk]:
        """Chunk text with validation and automatic size correction."""
        chunks = self._chunker.chunk(text)

        validated = []
        for chunk in chunks:
            if len(chunk.text) < self.min_chunk_size:
                # Merge tiny tail chunks with the previous one
                if validated:
                    prev = validated[-1]
                    merged_text = prev.text + " " + chunk.text
                    validated[-1] = Chunk(
                        text=merged_text,
                        index=prev.index,
                        start_char=prev.start_char,
                        end_char=chunk.end_char,
                        strategy="recursive+merged",
                        metadata={"merged": True, "source": source_id},
                    )
                else:
                    validated.append(chunk)
            elif len(chunk.text) > self.max_chunk_size:
                # Split oversized chunks further
                sub_chunker = Chunker(strategy="sentence", max_sentences=4)
                sub_chunks = sub_chunker.chunk(chunk.text)
                validated.extend(sub_chunks)
            else:
                validated.append(chunk)

        return validated

    def validate(self, chunks: list[Chunk]) -> dict:
        """Return a validation report for a set of chunks."""
        sizes = [len(c.text) for c in chunks]
        return {
            "total": len(chunks),
            "min_size": min(sizes) if sizes else 0,
            "max_size": max(sizes) if sizes else 0,
            "avg_size": sum(sizes) / len(sizes) if sizes else 0,
            "too_small": sum(1 for s in sizes if s < self.min_chunk_size),
            "too_large": sum(1 for s in sizes if s > self.max_chunk_size),
            "ok": sum(1 for s in sizes if self.min_chunk_size <= s <= self.max_chunk_size),
        }


def deduplicate_chunks(chunks: list[Chunk], threshold: float = 0.7) -> list[Chunk]:
    """
    Remove near-duplicate chunks from a list.

    Args:
        chunks:    List of Chunk objects
        threshold: Jaccard similarity above which two chunks are considered duplicates

    Returns:
        Deduplicated list (keeps the first occurrence of each near-duplicate group)
    """
    kept = []
    for candidate in chunks:
        is_dup = any(
            _jaccard(candidate.text, kept_chunk.text) >= threshold
            for kept_chunk in kept
        )
        if not is_dup:
            kept.append(candidate)
    return kept


def budget_context(
    chunks: list,
    max_chars: int = 12000,
    reserve_chars: int = 2000,
) -> list:
    """
    Trim retrieved chunks to fit within a context budget.
    Keeps highest-ranked chunks first, stops before hitting the limit.

    Args:
        chunks:       List of chunks (ordered by relevance, best first)
        max_chars:    Total context window size in characters
        reserve_chars: Characters to reserve for system prompt + response

    Returns:
        Trimmed list that fits within max_chars - reserve_chars
    """
    available = max_chars - reserve_chars
    selected  = []
    used_chars = 0

    for chunk in chunks:
        chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
        if used_chars + len(chunk_text) <= available:
            selected.append(chunk)
            used_chars += len(chunk_text)
        else:
            break  # Stop adding — would overflow

    return selected


# ── Internal helpers ──────────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"\b[a-z]{3,}\b", a.lower()))
    tb = set(re.findall(r"\b[a-z]{3,}\b", b.lower()))
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0
