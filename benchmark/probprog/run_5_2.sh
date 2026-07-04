#!/bin/bash
#
# Run all §5.2 benchmarks (Prior Work Comparison).
#
# Runs Impulse, NumPyro, and Stan on:
#   - 3 standard benchmarks (logistic regression, robust regression, N-Schools)
#   - 1 motivating example (GP Poisson regression)
#
# Usage:
#   ./run_5_2.sh                    # run and print table
#   ./run_5_2.sh --extract-only     # just extract from existing log
#
# Output:
#   Log:     /tmp/bench_section52.log
#   CSV:     /tmp/section52_results.csv
#   Table:   printed to stdout

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN="$SCRIPT_DIR/run_pplbench.sh"
EXTRACT="$SCRIPT_DIR/extract_results.py"
LOG="/tmp/bench_section52.log"
CSV="/tmp/section52_results.csv"
THREADS=32
PPLS="numpyro,impulse,stan"

BENCHMARKS=(
    standard/logistic_regression
    standard/robust_regression
    standard/n_schools
    motivating/gp_pois_regr
)

if [[ "${1:-}" == "--extract-only" ]]; then
    echo "Extracting from existing log: $LOG"
    python3 "$EXTRACT" "$LOG" --csv "$CSV"
    exit 0
fi

echo "=== §5.2 Benchmarks ==="
echo "PPLs: $PPLS"
echo "Threads: $THREADS"
echo "Log: $LOG"
echo ""

> "$LOG"  # truncate log

for bench in "${BENCHMARKS[@]}"; do
    echo "--- Running $bench ---"
    CUDA_VISIBLE_DEVICES="" "$RUN" --threads "$THREADS" --ppls "$PPLS" "$bench" 2>&1 | tee -a "$LOG"
done

echo ""
echo "=== Results ==="
python3 "$EXTRACT" "$LOG" --csv "$CSV"
