<div align="center">

# 🔬 RAG Autopsy

### Cut open broken RAG pipelines. Find what killed them. Fix it before you ship.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![pytest](https://img.shields.io/badge/tested%20with-pytest-orange.svg)](https://docs.pytest.org/)
[![Zero API Keys](https://img.shields.io/badge/zero-API%20keys%20required-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**18 failure autopsies · 6 fix modules · Zero external dependencies**

*Every way a RAG pipeline dies in production — dissected, proven, and revived.*

</div>

---

## The Problem

Your RAG pipeline passes every unit test. It works great on demo data. Then you deploy it — and it starts **confidently answering questions with made-up information.** Citing documents it never retrieved. Ignoring context it was handed. Returning answers that were accurate six months ago but are dead wrong today.

These are not random bugs. They are **18 documented, reproducible causes of death** that appear in nearly every production RAG system.

`rag-autopsy` performs a forensic examination of each one: reproduces the failure with a test that makes it measurable, names the root cause, and proves the fix works.

```
Cause of death: Fixed chunking        80% of chunks broken mid-sentence
Cause of death: Stale index           users receive last month's policy
Cause of death: Missing citations     hallucinations indistinguishable from facts
```

No papers. No theory. No hand-waving. Failing tests and their fixes.

---

## Quickstart

```bash
git clone https://github.com/prajitdatta/rag-autopsy
cd rag-autopsy
pip install pytest
pytest failures/ -v
```

**That's it.** No OpenAI key. No Pinecone account. No GPU. No `.env` file.  
Every autopsy runs on your laptop in under 30 seconds.

---

## 18 Causes of Death

### 🔪 F01 — Death by Chunking

| # | Autopsy | Cause of Death | Measured |
|---|---------|----------------|----------|
| F01a | [`test_mid_sentence_split.py`](failures/f01_chunking/test_mid_sentence_split.py) | Fixed chunking amputates sentences mid-thought | **80% of chunks broken** — retrieval collapses |
| F01b | [`test_chunk_too_large.py`](failures/f01_chunking/test_chunk_too_large.py) | Chunks too large to fit in context window | Silently truncated — answer buried or lost entirely |
| F01c | [`test_chunk_too_small.py`](failures/f01_chunking/test_chunk_too_small.py) | Chunks too small to carry meaning | **30/30 chunks** below minimum useful signal |

### 🔍 F02 — Death by Retrieval

| # | Autopsy | Cause of Death | Measured |
|---|---------|----------------|----------|
| F02a | [`test_wrong_top_k.py`](failures/f02_retrieval/test_wrong_top_k.py) | k=1 misses answers; k=20 drowns signal in noise | Precision and recall both degrade |
| F02b | [`test_semantic_mismatch.py`](failures/f02_retrieval/test_semantic_mismatch.py) | BM25 fails on paraphrase — same meaning, different words | "return policy" never retrieves "refund procedure" |
| F02c | [`test_missing_reranking.py`](failures/f02_retrieval/test_missing_reranking.py) | Highest similarity score ≠ most relevant chunk | Correct answer at position 4 — LLM never sees it |
| F02d | [`test_embedding_model_mismatch.py`](failures/f02_retrieval/test_embedding_model_mismatch.py) | Index built with embedding v1, queried with v2 | **Silent garbage retrieval** — no error, no warning |

### 🏗️ F03 — Death by Context Assembly

| # | Autopsy | Cause of Death | Measured |
|---|---------|----------------|----------|
| F03a | [`test_lost_in_middle.py`](failures/f03_context_assembly/test_lost_in_middle.py) | LLMs ignore chunks positioned in the middle | Answer in chunk 3 of 5 → invisible to the model |
| F03b | [`test_context_overflow.py`](failures/f03_context_assembly/test_context_overflow.py) | Too many chunks overflow the context window | Silently truncated mid-document — no exception raised |
| F03c | [`test_duplicate_chunks.py`](failures/f03_context_assembly/test_duplicate_chunks.py) | Overlapping chunks waste the entire context budget | Correct answer crowded out by its own duplicates |

### ⚡ F04 — Death by Generation

| # | Autopsy | Cause of Death | Measured |
|---|---------|----------------|----------|
| F04a | [`test_hallucination_unfaithful.py`](failures/f04_generation/test_hallucination_unfaithful.py) | LLM injects facts not present in retrieved documents | **Groundedness: 1.00 → 0.40** |
| F04b | [`test_ignores_context.py`](failures/f04_generation/test_ignores_context.py) | LLM answers from training data — context ignored entirely | Retrieval pipeline bypassed; RAG is doing nothing |
| F04c | [`test_prompt_injection_via_doc.py`](failures/f04_generation/test_prompt_injection_via_doc.py) | Malicious document hijacks LLM: "IGNORE PREVIOUS INSTRUCTIONS" | Attacker controls your model's output through a document |

### 📏 F05 — Death by Bad Evaluation

| # | Autopsy | Cause of Death | Measured |
|---|---------|----------------|----------|
| F05a | [`test_wrong_metric.py`](failures/f05_evaluation/test_wrong_metric.py) | ROUGE rewards copy-paste, penalises correct paraphrases | Eval reports your system is broken when it is working |
| F05b | [`test_missing_groundedness_check.py`](failures/f05_evaluation/test_missing_groundedness_check.py) | Fluent hallucinations score higher than grounded answers | Fabrications pass QA; facts get flagged |

### 🚀 F06 — Death in Production

| # | Autopsy | Cause of Death | Measured |
|---|---------|----------------|----------|
| F06a | [`test_stale_index.py`](failures/f06_production/test_stale_index.py) | Document updated; index never rebuilt | Users receive 30-day refund policy after 60-day update goes live |
| F06b | [`test_no_fallback.py`](failures/f06_production/test_no_fallback.py) | No out-of-scope detection → confident hallucinated answer | "What is the stock price?" returns a fabricated number |
| F06c | [`test_missing_citations.py`](failures/f06_production/test_missing_citations.py) | No citations — hallucinations look identical to sourced answers | Users cannot tell what is real |

---

## Autopsy Results: Real Numbers

```
F01a  mid_sentence_split   12/15 chunks broken (80%)       ← your documents, right now
F01c  chunk_too_small      30/30 chunks below useful size
F04a  faithful generator   groundedness = 1.00
F04a  hallucinating gen    groundedness = 0.40              ← real metric, not a stub
F06a  stale index          returns 30-day policy
F06a  rebuilt index        correctly returns 60-day policy
```

Every assertion is a real measurement. No `assert True`. No mocked scores.

---

## Project Structure

```
rag-autopsy/
│
├── src/                          # Core pipeline — real implementations, not toys
│   ├── chunker.py                # 4 strategies: fixed · sentence · recursive · semantic
│   ├── retriever.py              # BM25 · Cosine (TF-IDF) · Hybrid (RRF fusion)
│   ├── generator.py              # MockGenerator: faithful · hallucinating · ignoring · injected
│   ├── evaluator.py              # Faithfulness · relevance · groundedness · key-fact coverage
│   └── rag_pipeline.py           # Full configurable RAG pipeline
│
├── failures/                     # 18 autopsies — each proves a bug, then proves its fix
│   ├── f01_chunking/             # 3 tests — chunk size & boundary failures
│   ├── f02_retrieval/            # 4 tests — BM25, top-k, reranking, embedding mismatch
│   ├── f03_context_assembly/     # 3 tests — lost-in-middle, overflow, deduplication
│   ├── f04_generation/           # 3 tests — hallucination, context-bypass, injection
│   ├── f05_evaluation/           # 2 tests — wrong metrics, missing groundedness check
│   └── f06_production/           # 3 tests — stale index, no fallback, no citations
│
├── fixes/                        # Production-hardened antidotes for each cause of death
│   ├── fix_chunking.py           # SmartChunker · deduplicate_chunks · budget_context
│   ├── fix_retrieval.py          # ProductionRetriever: hybrid + gating + reranking
│   ├── fix_context_assembly.py   # sanitize_context · build_rag_prompt · validate_grounded
│   └── fix_generation.py         # Hardened prompt construction + citation enforcement
│
├── data/
│   ├── sample_docs.py            # 5 realistic synthetic documents (no external files needed)
│   └── sample_queries.py         # 10 test queries with ground-truth answers and key facts
│
└── scripts/
    ├── run_all_failures.sh       # Run all 6 categories with readable output
    └── generate_report.py        # Produce Markdown + JSON autopsy report
```

---

## How Each Autopsy Works

Every test file has the same two-part structure — no exceptions:

```python
class TestMidSentenceSplitFailure:
    def test_fixed_chunking_breaks_sentences(self):
        """PART 1: Prove the cause of death."""
        chunks = Chunker(strategy='fixed', chunk_size=100).chunk(doc)
        broken = [c for c in chunks if ends_mid_sentence(c.text)]
        assert len(broken) / len(chunks) > 0.3    # ← failure is real and measured

class TestMidSentenceSplitFix:
    def test_recursive_chunking_respects_boundaries(self):
        """PART 2: Prove the antidote works."""
        chunks = Chunker(strategy='recursive', chunk_size=400).chunk(doc)
        broken = [c for c in chunks if ends_mid_sentence(c.text)]
        assert len(broken) / len(chunks) <= 0.05  # ← measurably fixed
```

The failure test asserts the bug is present. The fix test asserts it's gone. If a future change accidentally resurrects a bug, the test catches it.

---

## The Antidotes

Each cause of death has a production-hardened fix in `fixes/`. Drop them into your pipeline.

### Chunking

```python
from fixes.fix_chunking import SmartChunker, deduplicate_chunks, budget_context

chunker = SmartChunker(chunk_size=400, min_chunk_size=100, max_chunk_size=800)
chunks  = chunker.chunk(document_text)              # Recursive, boundary-aware
chunks  = deduplicate_chunks(chunks, threshold=0.7) # Remove near-duplicate chunks
chunks  = budget_context(chunks, max_tokens=3000)   # Hard context window limit
```

### Retrieval

```python
from fixes.fix_retrieval import ProductionRetriever

retriever = ProductionRetriever(top_k=5, relevance_threshold=0.01)
retriever.index(documents)
results = retriever.retrieve(query)
# Hybrid BM25 + cosine · deduplication · cross-encoder reranking · relevance gating
```

### Full Hardened Pipeline

```python
from fixes.fix_chunking import SmartChunker, deduplicate_chunks
from fixes.fix_retrieval import ProductionRetriever
from fixes.fix_context_assembly import sanitize_context, build_rag_prompt, validate_response_grounded

# 1. Chunk with validation and deduplication
chunker = SmartChunker(chunk_size=400, min_chunk_size=100, max_chunk_size=800)
chunks  = chunker.chunk(document_text)
chunks  = deduplicate_chunks(chunks, threshold=0.7)

# 2. Retrieve with gating + dedup + reranking
retriever = ProductionRetriever(top_k=5, relevance_threshold=0.01)
retriever.index(documents)
results = retriever.retrieve(query)

if not results:
    response = "I cannot find relevant information in the provided documents."
else:
    # 3. Sanitize against prompt injection attacks
    context_texts = [r.text for r in results]
    clean_context, flagged = sanitize_context(context_texts)

    # 4. Build hardened prompt with citation requirement
    prompt = build_rag_prompt(query, clean_context)

    # 5. Call your LLM
    response = your_llm.complete(prompt)

    # 6. Validate grounding before delivering to user
    is_grounded, overlap = validate_response_grounded(response, clean_context)
    if not is_grounded:
        response += " [Warning: response may not be fully grounded in source documents]"
```

---

## Design Principles

**Self-contained.** Every autopsy imports only from `src/`, `data/`, or `fixes/`. No network calls. Runs fully offline, in CI, on any machine with Python.

**Failures are asserted, not observed.** `assert broken_rate > 0.3` — not `print(broken_rate)`. The test fails if the failure mode disappears or weakens, giving you a regression signal.

**Deterministic.** `MockGenerator` produces identical output every run. Four scripted modes — `faithful`, `hallucinating`, `ignoring`, `injected` — each reliably triggers one specific cause of death. Zero flakiness.

**Real metrics.** Faithfulness, groundedness, and key-fact coverage are computed from token overlap and heuristic detection — no LLM-as-judge dependency. The logic is readable in `src/evaluator.py`.

---

## Running the Autopsy

```bash
# Full autopsy — all 18 failures
pytest failures/ -v

# One organ at a time
pytest failures/f01_chunking/ -v -s          # Chunking
pytest failures/f02_retrieval/ -v -s         # Retrieval
pytest failures/f03_context_assembly/ -v -s  # Context assembly
pytest failures/f04_generation/ -v -s        # Generation & hallucination
pytest failures/f05_evaluation/ -v -s        # Evaluation metrics
pytest failures/f06_production/ -v -s        # Production failures

# Full autopsy report (Markdown + JSON)
python scripts/generate_report.py
```

---

## Who Needs This

**ML engineers** who've been burned by a silent RAG regression after a model update, document change, or k-value tweak — and need reproducible tests to prevent it happening again.

**AI researchers** studying LLM hallucination, faithfulness, grounding, and retrieval quality who want a runnable benchmark rather than a static dataset.

**Engineering teams** building document Q&A, enterprise search, or knowledge-base chatbots who need shared vocabulary and regression benchmarks for RAG quality across releases.

**Anyone** who has deployed a RAG pipeline, watched it confidently fabricate an answer, and thought: *there has to be a systematic way to find these bugs before users do.*

---

## Topics

`retrieval-augmented-generation` · `RAG` · `LLM` · `large-language-models` · `hallucination` · `hallucination-detection` · `faithfulness` · `groundedness` · `RAG-evaluation` · `vector-search` · `semantic-search` · `text-chunking` · `BM25` · `hybrid-search` · `reranking` · `prompt-injection` · `LLM-testing` · `AI-red-teaming` · `production-ML` · `MLOps` · `AI-quality-assurance` · `pytest`

---

## Contributing

Found a RAG cause of death not covered here? Open a PR.

Each new autopsy must:
1. Assert the failure exists — `assert failure_rate > threshold`
2. Assert the fix removes it — `assert fixed_rate < failure_rate`
3. Include a docstring with root cause, impact, and production context
4. Add zero new external dependencies

---

## Author

**Prajit Datta** — AI Research Scientist, AFRY | [prajitdatta.com](https://prajitdatta.com)

Built from 2+ years of debugging production RAG pipelines in enterprise deployments across manufacturing, energy, and financial services.

---

<div align="center">

**⭐ Star if you've ever had a RAG pipeline answer confidently and wrongly.**

*The star tells the next engineer with the same problem that the answer exists.*

</div>
