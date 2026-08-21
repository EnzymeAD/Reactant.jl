#!/bin/bash
#
# Run --validate (SVI variational correctness mode) across all three SVI PPLBench
# models and aggregate per-model per-quantity max_diff / PASS-FAIL into one JSON.
#
# Each model is run as:
#   CUDA_VISIBLE_DEVICES='' ./run_pplbench.sh --validate --threads N svi/<config>
# which keeps only the NumPyro + Impulse arms and (see
# pplbench/pplbench/main.py:_apply_validate_overrides_svi) forces a short
# deterministic AutoNormal/Adam fit (num_steps=20, step_size=0.01, init_scale=0.1,
# num_particles=1), 2 trials, matched loc=0 init, and a matched reparam-noise RNG
# stream so both arms are pointwise comparable on the FIT (fitted guide loc/scale
# + per-step ELBO). The harness writes a validation_summary.json into each run's
# output dir; this script collects them into validate_svi_results_<date>.json.
#
# This is the SVI half only; the NUTS half lives in run_all_validate.sh (kept
# intact). SVI validation compares the fitted guide, NOT the posterior draws in
# samples*.nc (those use a separate draw RNG and legitimately differ).
#
# Env overrides:
#   THREADS               intra-op threads for all frameworks (default 8)
#   VALIDATE_LOG_DIR      where per-model transcripts go (default ./validate_svi_transcripts)
#   VALIDATE_RESULTS_JSON aggregated results path (default ./validate_svi_results_<date>.json)
#   VALIDATE_TIMEOUT      per-model timeout in seconds (default 1800)
#   PYTHON                python for aggregation (default conda probprog-bench)

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP="$(date +%Y-%m-%d)"
THREADS="${THREADS:-8}"
LOG_DIR="${VALIDATE_LOG_DIR:-$SCRIPT_DIR/validate_svi_transcripts}"
RESULTS_JSON="${VALIDATE_RESULTS_JSON:-$SCRIPT_DIR/validate_svi_results_${TIMESTAMP}.json}"
PER_MODEL_TIMEOUT="${VALIDATE_TIMEOUT:-1800}"
PYTHON="${PYTHON:-/home/sbrantq/anaconda3/envs/probprog-bench/bin/python}"
mkdir -p "$LOG_DIR"

CONFIGS=(
    svi/logistic_regression
    svi/robust_regression
    svi/n_schools
)

MANIFEST="$LOG_DIR/summaries_manifest.txt"
: > "$MANIFEST"
FAILED=()

for config in "${CONFIGS[@]}"; do
    slug="${config//\//_}"
    transcript="$LOG_DIR/validate_${slug}.log"

    echo ""
    echo "====== VALIDATE SVI: $config (threads=$THREADS) ======"
    echo "  transcript: $transcript"

    cmd="CUDA_VISIBLE_DEVICES='' '$SCRIPT_DIR/run_pplbench.sh' --validate --threads $THREADS '$config'"
    if timeout "$PER_MODEL_TIMEOUT" script -q -c "$cmd" "$transcript"; then
        echo "====== $config: DONE ======"
    else
        echo "====== $config: FAIL (nonzero exit / timeout) ======"
        FAILED+=("$config")
    fi

    # The harness prints "Output dir: <path>" and writes validation_summary.json there.
    outdir="$(grep -a -m1 '^Output dir:' "$transcript" | sed 's/^Output dir: //' | tr -d '\r')"
    summary="$outdir/validation_summary.json"
    if [ -n "$outdir" ] && [ -f "$summary" ]; then
        printf '%s::%s\n' "$config" "$summary" >> "$MANIFEST"
        echo "  summary: $summary"
    else
        echo "  summary: MISSING (no validation_summary.json for $config)"
    fi
done

# Aggregate every per-model summary into one results JSON.
"$PYTHON" - "$RESULTS_JSON" "$MANIFEST" <<'PY'
import json, sys
out_path, manifest = sys.argv[1], sys.argv[2]
results = {}
with open(manifest) as m:
    for line in m:
        line = line.strip()
        if not line:
            continue
        config, path = line.split("::", 1)
        with open(path) as f:
            results[config] = json.load(f)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nAggregated {len(results)} model summaries -> {out_path}")
for cfg, s in results.items():
    tag = "PASS" if s.get("all_close") else "FAIL"
    worst = 0.0
    for seed_entry in s.get("per_seed", {}).values():
        for q in ("loc", "scale"):
            for site in seed_entry.get(q, {}).values():
                worst = max(worst, site.get("max_diff", 0.0))
        e = seed_entry.get("elbo")
        if e:
            worst = max(worst, e.get("max_diff", 0.0))
    print(f"  {cfg}: {tag}  worst_max_diff={worst:.6e}")
PY

echo ""
echo "=== Summary ==="
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED configs (nonzero exit / timeout):"
    for c in "${FAILED[@]}"; do echo "  - $c"; done
    exit 1
else
    echo "All SVI configs validated"
fi
