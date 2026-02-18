#!/usr/bin/env bash
# scripts/run_all_failures.sh
# Run the full failure test suite with a readable summary.

set -euo pipefail
cd "$(dirname "$0")/.."

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          RAG FAILURE MODES — Full Test Suite                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

run_group() {
    local label=$1
    local path=$2
    echo "────────────────────────────────────────────────────────────────"
    echo " $label"
    echo "────────────────────────────────────────────────────────────────"
    python -m pytest "$path" -v --tb=short --no-header 2>&1 || true
    echo ""
}

run_group "F01 — CHUNKING FAILURES"          "failures/f01_chunking/"
run_group "F02 — RETRIEVAL FAILURES"         "failures/f02_retrieval/"
run_group "F03 — CONTEXT ASSEMBLY FAILURES"  "failures/f03_context_assembly/"
run_group "F04 — GENERATION FAILURES"        "failures/f04_generation/"
run_group "F05 — EVALUATION FAILURES"        "failures/f05_evaluation/"
run_group "F06 — PRODUCTION FAILURES"        "failures/f06_production/"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " SUMMARY"
echo "════════════════════════════════════════════════════════════════"
python -m pytest failures/ --tb=no -q 2>&1 | tail -5
