#!/bin/bash
# Run all standard + SICM + motivating benchmarks sequentially.
# Uses `script -c` to capture full terminal output including control codes.
#
# Usage:
#   bash run_all_benchmarks.sh              # Run all benchmarks
#   bash run_all_benchmarks.sh standard     # Run only standard benchmarks
#   bash run_all_benchmarks.sh sicm         # Run only SICM benchmarks
#   bash run_all_benchmarks.sh motivating   # Run only motivating benchmarks

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP="$(date +%Y-%m-%d_%H:%M:%S)"
TRANSCRIPT_DIR="$SCRIPT_DIR/logs"
mkdir -p "$TRANSCRIPT_DIR"
MASTER_LOG="$TRANSCRIPT_DIR/benchmark_${TIMESTAMP}.log"

STANDARD_CONFIGS=(
    standard/logistic_regression
    standard/robust_regression
    standard/gaussian_process
    standard/n_schools
)

SICM_CONFIGS=(
    sicm/hierarchical_mvn
    sicm/scale_family_mvn
    sicm/scale_family_mvn_n100
    sicm/gp_pois_regr_scaled
    sicm/gp_classify
    sicm/gp_fixed_lengthscale
    sicm/gp_fixed_lengthscale_n100
    sicm/phylogenetic_regression
    sicm/linear_mixed_model
    sicm/linear_mixed_model_n200
    sicm/stochastic_volatility
    sicm/car_spatial
)

MOTIVATING_CONFIGS=(
    motivating/gp_pois_regr
)

# Select which configs to run
FILTER="${1:-all}"
CONFIGS=()
case "$FILTER" in
    standard)
        CONFIGS=("${STANDARD_CONFIGS[@]}")
        ;;
    sicm)
        CONFIGS=("${SICM_CONFIGS[@]}")
        ;;
    motivating)
        CONFIGS=("${MOTIVATING_CONFIGS[@]}")
        ;;
    all)
        CONFIGS=("${STANDARD_CONFIGS[@]}" "${MOTIVATING_CONFIGS[@]}" "${SICM_CONFIGS[@]}")
        ;;
    *)
        echo "Unknown filter: $FILTER (use: all, standard, sicm, motivating)"
        exit 1
        ;;
esac

echo "=== Starting benchmarks ($FILTER) at $(date) ===" | tee "$MASTER_LOG"
echo "Master log: $MASTER_LOG"
echo "Transcripts: $TRANSCRIPT_DIR/"
echo "Configs to run: ${#CONFIGS[@]}"
echo ""

FAILED=()
OUTPUT_DIRS=()

for config in "${CONFIGS[@]}"; do
    # Create per-config transcript filename (replace / with _)
    config_slug="${config//\//_}"
    transcript="$TRANSCRIPT_DIR/${config_slug}_${TIMESTAMP}.txt"

    echo "" | tee -a "$MASTER_LOG"
    echo "====== $config ($(date)) ======" | tee -a "$MASTER_LOG"
    echo "  transcript: $transcript" | tee -a "$MASTER_LOG"

    if script -c "bash '$SCRIPT_DIR/run_pplbench.sh' '$config'" "$transcript"; then
        echo "====== $config DONE ($(date)) ======" | tee -a "$MASTER_LOG"
    else
        echo "====== $config FAILED ($(date)) ======" | tee -a "$MASTER_LOG"
        FAILED+=("$config")
    fi
done

echo "" | tee -a "$MASTER_LOG"
echo "=== All benchmarks completed at $(date) ===" | tee -a "$MASTER_LOG"

# Find output directories for each config (most recent run)
echo "" | tee -a "$MASTER_LOG"
echo "=== Output directories ===" | tee -a "$MASTER_LOG"
for config in "${CONFIGS[@]}"; do
    config_file="$SCRIPT_DIR/pplbench_configs/${config}.json"
    model_class=$(python3 -c "import json; print(json.load(open('$config_file'))['model']['class'])" 2>/dev/null || echo "unknown")
    dir=$(ls -td "$SCRIPT_DIR"/outputs/*/config.json 2>/dev/null | while read f; do
        if grep -q "$model_class" "$f" 2>/dev/null; then echo "$(dirname "$f")"; break; fi
    done)
    echo "$config: $dir" | tee -a "$MASTER_LOG"
    [ -n "$dir" ] && OUTPUT_DIRS+=("$dir")
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "" | tee -a "$MASTER_LOG"
    echo "=== FAILED benchmarks ===" | tee -a "$MASTER_LOG"
    for config in "${FAILED[@]}"; do
        echo "  - $config" | tee -a "$MASTER_LOG"
    done
fi

echo "" | tee -a "$MASTER_LOG"
echo "=== Next steps ===" | tee -a "$MASTER_LOG"
if [ ${#OUTPUT_DIRS[@]} -gt 0 ]; then
    echo "  python collect_paper_data.py --from-outputs ${OUTPUT_DIRS[*]}" | tee -a "$MASTER_LOG"
fi
echo "  python generate_tables.py" | tee -a "$MASTER_LOG"
