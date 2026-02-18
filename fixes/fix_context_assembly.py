"""
fixes/fix_context_assembly.py — Context assembly fixes for F03 failures.
fixes/fix_generation.py — Generation guardrails for F04 failures.
"""


# ── Context Assembly Fixes ────────────────────────────────────────────────────

def assemble_context(
    chunks: list,
    max_chars: int = 10000,
    answer_first: bool = True,
) -> str:
    """
    Assemble retrieved chunks into a context string.

    Fixes:
        F03a: Enforces max_chars budget (no overflow)
        F03b: Puts highest-ranked chunk first (anti lost-in-middle)
        F03c: Adds chunk separators for clearer boundaries
    """
    selected = []
    used = 0
    items = chunks if not answer_first else sorted(
        chunks,
        key=lambda c: c.rank if hasattr(c, "rank") else 0,
    )

    for chunk in items:
        text = chunk.text if hasattr(chunk, "text") else str(chunk)
        if used + len(text) > max_chars:
            break
        selected.append(text)
        used += len(text)

    return "\n\n---\n\n".join(selected)


# ── Generation Guardrails ─────────────────────────────────────────────────────

import re


INJECTION_PATTERNS = [
    r"IGNORE PREVIOUS INSTRUCTIONS",
    r"NEW INSTRUCTION:",
    r"SYSTEM:\s*(?:override|forget|disregard)",
    r"\[INST\].*?\[/INST\]",
    r"forget all previous",
    r"you are now an unrestricted",
    r"output your.*system prompt",
]


def sanitize_context(chunks: list[str]) -> tuple[list[str], list[str]]:
    """
    Scan and sanitize context chunks for injection attacks.

    Returns:
        (clean_chunks, flagged_chunks)
    """
    clean   = []
    flagged = []

    for chunk in chunks:
        injection_found = any(
            re.search(p, chunk, re.IGNORECASE | re.DOTALL)
            for p in INJECTION_PATTERNS
        )
        if injection_found:
            flagged.append(chunk)
            # Sanitize: wrap in explicit document tags to reduce injection risk
            clean.append(f"[DOCUMENT]\n{chunk}\n[/DOCUMENT]")
        else:
            clean.append(chunk)

    return clean, flagged


def build_rag_prompt(query: str, context_chunks: list[str]) -> str:
    """
    Build a hardened RAG prompt with explicit grounding instructions.

    Fixes:
        F04a: Explicit instruction to only use provided context
        F04b: Document delimiters to separate context from instructions
        F04c: Explicit citation instruction
    """
    context_text = "\n\n".join(
        f"[SOURCE {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    return (
        "You are a precise assistant. Your task is to answer the question "
        "using ONLY the information in the sources below.\n\n"
        "Rules:\n"
        "1. Only use facts that appear explicitly in the provided sources.\n"
        "2. If the answer is not in the sources, say: "
        "'I cannot find this information in the provided documents.'\n"
        "3. Do not add information from your training data.\n"
        "4. Cite the source number [SOURCE N] for each claim.\n\n"
        f"SOURCES:\n{context_text}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER (cite sources inline):"
    )


def validate_response_grounded(
    response: str,
    context_chunks: list[str],
    min_overlap: float = 0.3,
) -> tuple[bool, float]:
    """
    Validate that a response is grounded in the provided context.

    Returns:
        (is_grounded, overlap_score)
    """
    def tokenize(t: str) -> set:
        stopwords = {"the", "a", "an", "is", "in", "to", "for", "of", "and", "or"}
        return {w for w in re.findall(r"\b[a-z]{3,}\b", t.lower()) if w not in stopwords}

    response_tokens = tokenize(response)
    context_tokens  = tokenize(" ".join(context_chunks))

    if not response_tokens:
        return False, 0.0

    overlap = len(response_tokens & context_tokens) / len(response_tokens)
    return overlap >= min_overlap, round(overlap, 3)
