#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP="$(date +%Y-%m-%d_%H:%M:%S)"
TRANSCRIPT_DIR="$SCRIPT_DIR/transcripts"
mkdir -p "$TRANSCRIPT_DIR"

CONFIGS=(
    standard/logistic_regression
    standard/robust_regression
    standard/gaussian_process
    standard/n_schools
    motivating/gp_pois_regr
    sicm/hierarchical_mvn
    sicm/scale_family_mvn
    sicm/gp_pois_regr_scaled
    sicm/gp_classify
    sicm/gp_fixed_lengthscale
    sicm/phylogenetic_regression
    sicm/linear_mixed_model
    sicm/stochastic_volatility
    sicm/car_spatial
)

FAILED=()
for config in "${CONFIGS[@]}"; do
    config_slug="${config//\//_}"
    transcript="$TRANSCRIPT_DIR/validate_${config_slug}_${TIMESTAMP}.txt"

    echo ""
    echo "====== VALIDATE: $config ======"
    echo "  transcript: $transcript"

    if script -c "bash '$SCRIPT_DIR/run_pplbench.sh' --validate '$config'" "$transcript"; then
        echo "====== $config: DONE ======"
    else
        echo "====== $config: FAIL ======"
        FAILED+=("$config")
    fi
done

echo ""
echo "=== Summary ==="
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED configs:"
    for c in "${FAILED[@]}"; do echo "  - $c"; done
else
    echo "All configs completed"
fi
