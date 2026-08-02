#!/bin/bash
#
# Run --validate (NUTS correctness mode) across all five NUTS PPLBench models
# and aggregate per-model per-variable max_diff / PASS-FAIL into one JSON.
#
# Each model is run as:
#   CUDA_VISIBLE_DEVICES='' ./run_pplbench.sh --validate --threads N <config>
# which keeps only the NumPyro + Impulse arms and (see
# pplbench/pplbench/main.py:_apply_validate_overrides_mcmc) forces 5 samples,
# 0 warmup, no adaptation, 2 trials, and matched initialization so both arms
# start from the same canonical constrained point. The harness writes a
# validation_summary.json into each run's output dir; this script collects them
# into validate_nuts_results_<date>.json.
#
# SVI configs are intentionally NOT covered here: the harness raises
# NotImplementedError for SVI validate (owned by the separate SVI validate
# agent). This script is the NUTS half only.
#
# Env overrides:
#   THREADS               intra-op threads for all frameworks (default 8)
#   VALIDATE_LOG_DIR      where per-model transcripts go (default ./validate_transcripts)
#   VALIDATE_RESULTS_JSON aggregated results path (default ./validate_nuts_results_<date>.json)
#   VALIDATE_TIMEOUT      per-model timeout in seconds (default 1800)
#   PYTHON                python for aggregation (default conda probprog-bench)

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP="$(date +%Y-%m-%d)"
THREADS="${THREADS:-8}"
LOG_DIR="${VALIDATE_LOG_DIR:-$SCRIPT_DIR/validate_transcripts}"
RESULTS_JSON="${VALIDATE_RESULTS_JSON:-$SCRIPT_DIR/validate_nuts_results_${TIMESTAMP}.json}"
PER_MODEL_TIMEOUT="${VALIDATE_TIMEOUT:-1800}"
PYTHON="${PYTHON:-/home/sbrantq/anaconda3/envs/probprog-bench/bin/python}"
mkdir -p "$LOG_DIR"

CONFIGS=(
    standard/logistic_regression
    standard/robust_regression
    standard/n_schools
    standard/crowd_sourced_annotation
    standard/noisy_or_topic
)

MANIFEST="$LOG_DIR/summaries_manifest.txt"
: > "$MANIFEST"
FAILED=()

for config in "${CONFIGS[@]}"; do
    slug="${config//\//_}"
    transcript="$LOG_DIR/validate_${slug}.log"

    echo ""
    echo "====== VALIDATE: $config (threads=$THREADS) ======"
    echo "  transcript: $transcript"

    cmd="CUDA_VISIBLE_DEVICES='' '$SCRIPT_DIR/run_pplbench.sh' --validate --threads $THREADS '$config'"
    # `-e` is REQUIRED: without it `script` always exits 0 and swallows the
    # harness's verdict, so a tolerance failure (exit 1) or a frozen/degenerate
    # chain (exit 2) would be silently reported as DONE.
    if timeout "$PER_MODEL_TIMEOUT" script -q -e -c "$cmd" "$transcript"; then
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
    vars_ = s.get("variables", {})
    worst = max((v["max_diff"] for v in vars_.values()), default=float("nan"))
    # Weakest movement across variables/arms: 0 means a frozen chain, which makes
    # the whole comparison vacuous no matter how small worst_max_diff is.
    moves = [x for v in vars_.values() for x in v.get("std_over_draws", {}).values()]
    weakest = min(moves) if moves else float("nan")
    degen = s.get("degenerate_variables") or []
    tag = "PASS" if s.get("all_close") else "FAIL"
    if degen:
        tag = "DEGENERATE"
    print(f"  {cfg}: {tag}  worst_max_diff={worst:.6e}  "
          f"weakest_std_over_draws={weakest:.6e}"
          + (f"  FROZEN: {','.join(degen)}" if degen else ""))
PY

echo ""
echo "=== Summary ==="
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED configs (nonzero exit / timeout):"
    for c in "${FAILED[@]}"; do echo "  - $c"; done
    exit 1
else
    echo "All configs validated"
fi
