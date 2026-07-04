#!/bin/bash
#
# Run ablation study benchmarks.
#
# Usage:
#   ./run_ablation.sh                    # Run all models
#   ./run_ablation.sh logistic_regression  # Run single model
#   ./run_ablation.sh --device cuda      # Run on GPU
#
# Each model runs Impulse in 3 configurations:
#   Full      - all optimizations (SICM + region inlining + canon/cse)
#   No SICM   - SICM disabled, region inlining + canon/cse still active
#   No region - region inlining skipped, model stays as func.call

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"
DEVICE="cpu"
SINGLE_MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            SINGLE_MODEL="$1"
            shift
            ;;
    esac
done

MODELS=(
    logistic_regression
    robust_regression
    n_schools
    gp_pois_regr_scaled
    gp_fixed_lengthscale
    linear_mixed_model
    stochastic_volatility
)

if [ -n "$SINGLE_MODEL" ]; then
    MODELS=("$SINGLE_MODEL")
fi

echo "=== Ablation Study ==="
echo "Device: $DEVICE"
echo "Models: ${MODELS[*]}"
echo ""

FAILED=()
for model in "${MODELS[@]}"; do
    echo "──────────────────────────────────────────────"
    echo "  Ablation: $model"
    echo "──────────────────────────────────────────────"
    if "$BENCHMARK_DIR/run_pplbench.sh" --device "$DEVICE" "ablation/$model"; then
        echo "  ✓ $model done"
    else
        echo "  ✗ $model FAILED"
        FAILED+=("$model")
    fi
    echo ""
done

echo "=== Ablation Complete ==="
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed: ${FAILED[*]}"
    exit 1
else
    echo "All models succeeded."
fi
