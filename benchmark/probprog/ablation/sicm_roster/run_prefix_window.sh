#!/bin/bash
# Re-measure, over its matched window only, every row whose two arms decorrelate
# before the end of the 20-draw window.
#
#   run_prefix_window.sh <batch_tag>
#
# Under the frozen-adapted protocol both arms start from one shared adapted
# state with matched seeds, so they take the same trajectory until a
# floating-point difference at the ULP floor has amplified enough to change a
# decision. `arm_prefix.py` reports how many leading draws still agree at or
# below 1e-8 relative. For a row whose prefix is shorter than the full window,
# the 20-draw ratio mixes matched and decorrelated draws; this script re-runs
# both arms with `num_samples` set to the prefix length, so the whole measured
# window is matched and the ratio is a per-step cost ratio again.
#
# Both numbers are then in the ledger: the full-window ratio from the main batch
# and the prefix-window ratio from tag `<batch_tag>_prefix`. Rows that already
# match over all 20 draws are skipped (their two numbers coincide).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TAG="${1:-refreeze_2026-09-08}"
LOGS="$HERE/logs_frozen/$TAG"
OUT="$LOGS/agreement"
mkdir -p "$OUT"

# model_key : julia_model_path : step_size : max_tree_depth  (same as the batch)
declare -A SPEC=(
  [car_spatial_scotland]="sicm/car_spatial:0.05:10"
  [car_spatial_nyc]="sicm/car_spatial:0.05:10"
  [pgls]="sicm/pgls:0.05:10"
  [hierarchical_mvn]="sicm/hierarchical_mvn:0.01:10"
  [lmm_radon]="sicm/linear_mixed_model:0.05:10"
  [sar_columbus]="sicm/sar_lag:0.05:10"
  [diamonds]="sicm/diamonds_regression:0.05:10"
  [phylo_pois]="motivating/phylo_pois:0.1:10"
)

CELLS=""
echo "== matched prefixes in batch $TAG"
for s in "$LOGS"/*_sicm.out.json; do
  [ -e "$s" ] || continue
  base="$(basename "$s" _sicm.out.json)"
  n="$LOGS/${base}_nosicm.out.json"
  [ -e "$n" ] || continue
  key="${base%%_seed*}"
  python3 "$HERE/arm_prefix.py" "$s" "$n" --json "$OUT/${base}.json" > /dev/null
  read -r pref tot <<< "$(python3 -c "import json,sys; d=json.load(open('$OUT/${base}.json')); print(d['matched_prefix'], d['n_draws'])")"
  echo "   $key: matched $pref of $tot draws"
  if [ "$pref" -lt "$tot" ] && [ "$pref" -ge 2 ]; then
    CELLS="$CELLS $key:${SPEC[$key]}:$pref"
  fi
done

if [ -z "$CELLS" ]; then
  echo "== every row matches over its whole window; no prefix re-run needed"
  exit 0
fi

echo "== rows needing a prefix-window re-run:$CELLS"
for c in $CELLS; do
  key="${c%%:*}"; rest="${c#*:}"
  model="${rest%%:*}"; rest="${rest#*:}"
  step="${rest%%:*}"; rest="${rest#*:}"
  depth="${rest%%:*}"; pref="${rest##*:}"
  echo "== $key over its matched window of $pref draws"
  FROZEN_CELLS="$key:$model:$step:$depth" FROZEN_NUM_SAMPLES="$pref" \
    bash "$HERE/run_frozen_batch.sh" "${TAG}_prefix"
done

echo "== summarize the prefix-window batch"
FROZEN_SUMMARY_OUT="$HERE/frozen_state_prefix_RAW_2026-09-08.json" \
  python3 "$HERE/summarize_frozen.py" "${TAG}_prefix"
