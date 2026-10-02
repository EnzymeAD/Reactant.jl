#!/bin/bash
# The SlicStan comparison (figure "slicstan-ladder"), measured as ONE GATED CELL
# PER PROGRAM VARIANT instead of one cell per system.
#
# Protocol unchanged: same model (gp_pois_regr_scaled, n=200, rho=0.15, seed
# 12345), same 500 iterations / 200 warmup / 4 trials, same per-variant seed
# 20260822, same infer_args, same single-thread pin, same gates. The seven
# configs under pplbench_configs/related_work/ are the source config split into
# one ppl entry each, verified field-for-field identical to it.
#
# Why: the two original cells (four Stan programs in one, three Impulse
# configurations in the other) are two-hour and one-hour windows, and this
# machine's quiet windows are minutes long, so neither cell ever fit one. The
# longest single variant is Stan naive at about 50 minutes for four trials and
# the rest are shorter, so a per-variant cell can be accepted where the
# aggregate cell cannot. NO GATE IS RELAXED: every cell runs through
# refreeze_cell.sh with the same LOAD_GATE / START_EXT_CPU / LOAD_MAX /
# EXT_CPU_LIMIT / ABORT_SAMPLES the chain script used for the aggregate cells.
#
# One cell per free CCD at a time, each pinned to one core on its own CCD; the
# remaining cells start as those finish. Longest variant first, so the tail of
# the pass is its cheapest cell.
#
# CORES defaults to "48 56" (CCD 6 and CCD 7), the CCDs that are actually free
# while this pass starts: CCD 0-3 carry the cross-framework sweep, CCD 8-11 the
# noisy-or w1000 row, and CCD 4 (core 32) and CCD 5 (core 40) are still held by
# the two aggregate ladder cells. Set CORES="32 40 48 56" once those two end, to
# run four at a time. One job per CCD is the campaign's pinning rule and is not
# relaxed here.
set -uo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
cd "$B" || exit 1
LOG="${1:-$B/logs/refreeze_2026-09-08/fig12}"
mkdir -p "$LOG/cells"

G="LOAD_GATE=${LOAD_GATE:-20} START_EXT_CPU=${START_EXT_CPU:-1000} LOAD_MAX=${LOAD_MAX:-24} \
EXT_CPU_LIMIT=${EXT_CPU_LIMIT:-1000} ABORT_SAMPLES=${ABORT_SAMPLES:-24} MAX_RETRIES=${MAX_RETRIES:-200}"

QUEUE="$LOG/.variant_queue"
LOCK="$LOG/.variant_queue.lock"
rm -rf "$LOCK"
cat > "$QUEUE" <<'EOQ'
ladder_stan_naive stan gp_pois_regr_ladder_4trials_stan_naive 4h
ladder_stan_slicstan stan gp_pois_regr_ladder_4trials_stan_slicstan 3h
ladder_stan_slicstan_local stan gp_pois_regr_ladder_4trials_stan_slicstan_local 3h
ladder_stan_expert stan gp_pois_regr_ladder_4trials_stan_expert 3h
ladder_impulse_no_preopt impulse gp_pois_regr_ladder_4trials_impulse_no_preopt 3h
ladder_impulse_std_passes impulse gp_pois_regr_ladder_4trials_impulse_std_passes 2h
ladder_impulse_sicm impulse gp_pois_regr_ladder_4trials_impulse_sicm 2h
EOQ
# VARIANT_FILTER (2026-09-22): keep only matching queue lines, e.g. "^ladder_impulse_".
if [ -n "${VARIANT_FILTER:-}" ]; then grep -E "$VARIANT_FILTER" "$QUEUE" > "$QUEUE.f"; mv "$QUEUE.f" "$QUEUE"; fi

pop() {
  local line
  while ! mkdir "$LOCK" 2>/dev/null; do sleep 1; done
  line="$(head -1 "$QUEUE" 2>/dev/null)"
  [ -n "$line" ] && sed -i '1d' "$QUEUE"
  rmdir "$LOCK"
  printf '%s' "$line"
}

worker() {
  local core="$1" line label ppl cfg tmo rc
  while :; do
    line="$(pop)"
    [ -z "$line" ] && break
    set -- $line
    label="$1"; ppl="$2"; cfg="$3"; tmo="$4"
    echo "[$(date '+%F %T')] core $core START $label ($cfg, timeout $tmo)"
    env $G ./refreeze_cell.sh "$label" "$LOG" "$tmo" none -- \
      "PPLBENCH_TASKSET_CPUS=$core ./run_pplbench.sh --threads 1 --ppls $ppl related_work/$cfg" \
      > "$LOG/$label.driver.log" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] core $core DONE  $label rc=$rc verdict=$(grep -h '^verdict=' "$LOG/cells/$label.gate.txt" 2>/dev/null | tail -1)"
  done
}

CORES="${CORES:-48 56}"
echo "[$(date '+%F %T')] SlicStan per-variant pass: 7 cells, cores $CORES, gates: $G"
for c in $CORES; do worker "$c" & done
wait
echo "[$(date '+%F %T')] all seven SlicStan variant cells done"
grep -H '^verdict=' "$LOG"/cells/ladder_stan_*.gate.txt "$LOG"/cells/ladder_impulse_*.gate.txt 2>/dev/null
