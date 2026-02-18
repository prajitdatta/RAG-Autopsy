"""
scripts/generate_report.py

Run all failure tests and generate a structured JSON + Markdown report.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --output report.md
"""

import subprocess
import json
import re
import sys
from pathlib import Path
from datetime import datetime


FAILURE_GROUPS = {
    "f01_chunking":           "F01 — Chunking",
    "f02_retrieval":          "F02 — Retrieval",
    "f03_context_assembly":   "F03 — Context Assembly",
    "f04_generation":         "F04 — Generation",
    "f05_evaluation":         "F05 — Evaluation",
    "f06_production":         "F06 — Production",
}


def run_tests(path: str) -> dict:
    """Run pytest on a path and parse results."""
    result = subprocess.run(
        ["python", "-m", "pytest", path, "--tb=no", "-q", "--no-header"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    output = result.stdout + result.stderr

    # Parse summary line: "X passed, Y failed in Zs"
    match = re.search(r"(\d+) passed(?:, (\d+) failed)?", output)
    passed = int(match.group(1)) if match else 0
    failed = int(match.group(2)) if (match and match.group(2)) else 0

    return {
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
        "output": output[:2000],
    }


def generate_report(output_file: str | None = None):
    """Run all tests and generate the report."""
    print("Running all failure tests...")
    print("=" * 60)

    results = {}
    total_passed = 0
    total_failed = 0

    for folder, label in FAILURE_GROUPS.items():
        path = f"failures/{folder}/"
        r = run_tests(path)
        results[folder] = {**r, "label": label}
        total_passed += r["passed"]
        total_failed += r["failed"]

        status = "✅" if r["failed"] == 0 else "⚠️ "
        print(f"  {status} {label}: {r['passed']}/{r['total']} passed")

    print(f"\n  Total: {total_passed}/{total_passed + total_failed} tests passed")

    # Build markdown report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# RAG Autopsy — Test Report",
        f"",
        f"Generated: {timestamp}",
        f"",
        f"## Summary",
        f"",
        f"| Category | Tests | Passed | Failed |",
        f"|----------|-------|--------|--------|",
    ]

    for folder, r in results.items():
        status = "✅" if r["failed"] == 0 else "❌"
        lines.append(f"| {status} {r['label']} | {r['total']} | {r['passed']} | {r['failed']} |")

    lines += [
        f"| **TOTAL** | **{total_passed + total_failed}** | **{total_passed}** | **{total_failed}** |",
        f"",
        f"## Failure Categories Tested",
        f"",
        f"| ID | Category | File |",
        f"|----|----------|------|",
        f"| F01a | Mid-sentence chunking splits | `failures/f01_chunking/test_mid_sentence_split.py` |",
        f"| F01b | Chunk size too large | `failures/f01_chunking/test_chunk_too_large.py` |",
        f"| F01c | Chunk size too small | `failures/f01_chunking/test_chunk_too_small.py` |",
        f"| F02a | Wrong top-k | `failures/f02_retrieval/test_wrong_top_k.py` |",
        f"| F02b | Semantic vocabulary mismatch | `failures/f02_retrieval/test_semantic_mismatch.py` |",
        f"| F02c | Missing reranking | `failures/f02_retrieval/test_missing_reranking.py` |",
        f"| F02d | Embedding model mismatch | `failures/f02_retrieval/test_embedding_model_mismatch.py` |",
        f"| F03a | Lost in the middle | `failures/f03_context_assembly/test_lost_in_middle.py` |",
        f"| F03b | Context overflow | `failures/f03_context_assembly/test_context_overflow.py` |",
        f"| F03c | Duplicate chunks | `failures/f03_context_assembly/test_duplicate_chunks.py` |",
        f"| F04a | Hallucination / unfaithful | `failures/f04_generation/test_hallucination_unfaithful.py` |",
        f"| F04b | LLM ignores context | `failures/f04_generation/test_ignores_context.py` |",
        f"| F04c | Prompt injection via document | `failures/f04_generation/test_prompt_injection_via_doc.py` |",
        f"| F05a | Wrong evaluation metric | `failures/f05_evaluation/test_wrong_metric.py` |",
        f"| F05b | Missing groundedness check | `failures/f05_evaluation/test_missing_groundedness_check.py` |",
        f"| F06a | Stale index | `failures/f06_production/test_stale_index.py` |",
        f"| F06b | No fallback for out-of-scope | `failures/f06_production/test_no_fallback.py` |",
        f"| F06c | Missing citations | `failures/f06_production/test_missing_citations.py` |",
    ]

    report_md = "\n".join(lines)

    if output_file:
        Path(output_file).write_text(report_md)
        print(f"\nReport written to: {output_file}")
    else:
        print(f"\n{'-'*60}")
        print(report_md)

    # Also save JSON
    json_report = {
        "timestamp": timestamp,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "groups": results,
    }
    Path("report.json").write_text(json.dumps(json_report, indent=2))
    print(f"JSON report: report.json")


if __name__ == "__main__":
    output = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--output" else None
    generate_report(output)
