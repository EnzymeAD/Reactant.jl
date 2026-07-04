#!/bin/bash
#
# Run all §5.3 SICM ablation benchmarks (Table 2).
#
# Compares SICM enabled vs disabled (disable_sicm flag) on 7 models,
# each targeting a specific SICM pattern.
#
# Usage:
#   ./run_5_3_sicm.sh                  # run all benchmarks
#   ./run_5_3_sicm.sh --extract-only   # just extract from existing log
#
# Output:
#   Log:     /tmp/bench_sicm_ablation.log
#   Results: /tmp/sicm_ablation_results.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN="$SCRIPT_DIR/run_pplbench.sh"
LOG="/tmp/bench_sicm_ablation.log"
RESULTS="/tmp/sicm_ablation_results.txt"
THREADS=32

# Pattern -> config mapping
declare -A PATTERN_NAMES
PATTERN_NAMES[sicm_ablation/gp_pois_regr]="CholScale"
PATTERN_NAMES[sicm_ablation/gp_classify]="CholScale"
PATTERN_NAMES[sicm_ablation/stochastic_volatility]="CholScale"
PATTERN_NAMES[sicm_ablation/scaled_solve]="TriSolveScale"
PATTERN_NAMES[sicm_ablation/scaled_multi_sensor]="DotScale"
PATTERN_NAMES[sicm_ablation/hierarchical_mvn]="OuterProdScale"
PATTERN_NAMES[sicm_ablation/eigenlift_solve]="EigenLift"

BENCHMARKS=(
    sicm_ablation/gp_pois_regr
    sicm_ablation/gp_classify
    sicm_ablation/stochastic_volatility
    sicm_ablation/scaled_solve
    sicm_ablation/scaled_multi_sensor
    sicm_ablation/hierarchical_mvn
    sicm_ablation/eigenlift_solve
)

extract_results() {
    local log="$1"
    local out="$2"

    echo "# SICM Ablation Results (Table 2)" > "$out"
    echo "# Generated: $(date)" >> "$out"
    echo "# Log: $log" >> "$out"
    echo "#" >> "$out"
    printf "%-20s %-22s %10s %10s %8s\n" "Pattern" "Model" "SICM (s)" "NoSICM (s)" "Speedup" >> "$out"
    printf "%-20s %-22s %10s %10s %8s\n" "-------" "-----" "--------" "----------" "-------" >> "$out"

    local current_model=""
    local current_ppl=""
    local sicm_time=""
    local nosicm_time=""
    local runtimes=()

    while IFS= read -r line; do
        # Detect model
        if [[ "$line" =~ "=== PPLBench: "(.+)" ===" ]]; then
            current_model="${BASH_REMATCH[1]}"
        fi

        # Detect PPL
        if [[ "$line" =~ "Starting inference on \`"(.+)"\`" ]]; then
            # Save previous PPL
            if [[ -n "$current_ppl" && ${#runtimes[@]} -gt 0 ]]; then
                # Use last trial as steady-state
                local steady="${runtimes[-1]}"
                if [[ "$current_ppl" == "SICM" ]]; then
                    sicm_time="$steady"
                elif [[ "$current_ppl" == "No SICM" ]]; then
                    nosicm_time="$steady"
                fi
            fi
            current_ppl="${BASH_REMATCH[1]}"
            runtimes=()
        fi

        # Parse runtime
        if [[ "$line" =~ "Subprocess compile="[0-9.]+"s, run="([0-9.]+)"s" ]]; then
            runtimes+=("${BASH_REMATCH[1]}")
        fi

        # At "Results appended" or "Trials completed for No SICM", flush
        if [[ "$line" =~ "Results appended" ]]; then
            # Save last PPL
            if [[ -n "$current_ppl" && ${#runtimes[@]} -gt 0 ]]; then
                local steady="${runtimes[-1]}"
                if [[ "$current_ppl" == "SICM" ]]; then
                    sicm_time="$steady"
                elif [[ "$current_ppl" == "No SICM" ]]; then
                    nosicm_time="$steady"
                fi
            fi

            if [[ -n "$sicm_time" && -n "$nosicm_time" ]]; then
                local model_short="${current_model##*/}"
                local pattern="${PATTERN_NAMES[$current_model]:-unknown}"
                local speedup=$(python3 -c "print(f'{$nosicm_time/$sicm_time:.1f}x')")
                printf "%-20s %-22s %10s %10s %8s\n" "$pattern" "$model_short" "$sicm_time" "$nosicm_time" "$speedup" >> "$out"
            fi
            sicm_time=""
            nosicm_time=""
            current_ppl=""
            runtimes=()
        fi
    done < "$log"

    echo "" >> "$out"
    cat "$out"
}

if [[ "${1:-}" == "--extract-only" ]]; then
    extract_results "$LOG" "$RESULTS"
    exit 0
fi

echo "=== §5.3 SICM Ablation (Table 2) ==="
echo "Benchmarks: ${#BENCHMARKS[@]}"
echo "Threads: $THREADS"
echo "Log: $LOG"
echo ""

> "$LOG"

for bench in "${BENCHMARKS[@]}"; do
    echo "--- Running $bench ---"
    CUDA_VISIBLE_DEVICES="" "$RUN" --threads "$THREADS" --ppls impulse "$bench" 2>&1 | tee -a "$LOG"
    echo ""
done

echo "=== Extracting Results ==="
extract_results "$LOG" "$RESULTS"
