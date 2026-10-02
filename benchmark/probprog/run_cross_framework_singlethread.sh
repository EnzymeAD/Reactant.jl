#!/bin/bash
#
# Cross-framework single-thread pinned sweep (paper figure "baselines" / fig 11).
#
# Protocol (2026-09-08 re-freeze):
#   * every cell single-threaded, taskset-pinned to one physical core on its own
#     CCD (8 cores/CCD on this Threadripper 7995WX; SMT sibling left idle),
#   * the systems of one model row run CONCURRENTLY so they share identical
#     ambient machine conditions; rows run sequentially,
#   * ONE replicate per cell for EVERY system (no per-system replicate
#     asymmetry). Flagged cells are RERUN, never averaged,
#   * matched deterministic initialization, 4000 iterations / 1000 warmup /
#     4 trials-as-chains, dense mass, Turing adtype per model (from the config).
#
# Gates, per cell:
#   * before the row: load1 < LOAD_GATE and external (non-$USER) CPU <
#     EXT_CPU_LIMIT, no sibling measurement of ours outside this sweep,
#   * during every cell: load1 and external CPU sampled every 5 s; a cell whose
#     window saw external CPU >= EXT_CPU_LIMIT or load1 > LOAD_MAX is
#     CONTAMINATED and the whole row is rerun (rows must share conditions),
#   * binary md5 verified before and after every row.
#
# Output layout is the one benchmark/probprog/extract_rerun_tables.py consumes:
#   driver.log        SPAWN/END lines with timestamps and cores
#   <label>.log       per-cell transcript (carries "Output dir:")
#   load_samples.csv  epoch,load1,extcpu  (whole-sweep sampling)
#   cells/<label>.gate.txt   per-cell load1/extcpu min-max-samples
#
# Env knobs: LOAD_GATE (24), LOAD_MAX (24), EXT_CPU_LIMIT (1000),
#            MAX_ROW_RETRIES (3), ROWS (space-separated row ids), LOGDIR.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

ME="$(id -un)"
LOAD_GATE="${LOAD_GATE:-24}"
LOAD_MAX="${LOAD_MAX:-24}"
EXT_CPU_LIMIT="${EXT_CPU_LIMIT:-1000}"
MAX_ROW_RETRIES="${MAX_ROW_RETRIES:-4}"
LOGDIR="${LOGDIR:-$SCRIPT_DIR/logs/refreeze_2026-09-08/fig11}"
SO="$SCRIPT_DIR/../../deps/ReactantExtra/bazel-bin/libReactantExtra.so"
mkdir -p "$LOGDIR/cells"
DRIVER_LOG="$LOGDIR/driver.log"
LOAD_CSV="$LOGDIR/load_samples.csv"
[ -f "$LOAD_CSV" ] || echo "epoch,load1,extcpu,our_running,ambient_load1" > "$LOAD_CSV"

note() { echo "[$(date '+%F %T')] $*" | tee -a "$DRIVER_LOG"; }
so_md5() { md5sum "$SO" 2>/dev/null | awk '{print $1}'; }
external_cpu() {
  ps -eo user,pcpu --no-headers 2>/dev/null \
    | awk -v me="$ME" '$1 != me { s += $2 } END { printf "%.0f", s+0 }'
}
load1() { cut -d' ' -f1 /proc/loadavg; }
# Concurrent pinned cells of this campaign legitimately raise load1, so the
# ambient (contamination) load is load1 minus our own runnable process count.
our_running() { ps -u "$ME" -o stat= --no-headers 2>/dev/null | grep -c '^R' || true; }
ambient_load1() { awk -v a="$(load1)" -v b="$(our_running)" 'BEGIN{printf "%.2f",(a-b<0?0:a-b)}'; }

BASE_MD5="$(so_md5)"
note "binary md5 = $BASE_MD5"
note "gates: load1 < $LOAD_GATE to start, load1 <= $LOAD_MAX during, external CPU < ${EXT_CPU_LIMIT}%"

# Whole-sweep load sampler (5 s), also feeds extract_rerun_tables.py.
( while :; do
    l="$(load1)"; o="$(our_running)"
    echo "$(date +%s),$l,$(external_cpu),$o,$(awk -v a="$l" -v b="$o" 'BEGIN{printf "%.2f",(a-b<0?0:a-b)}')" >> "$LOAD_CSV"
    sleep 5
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

# One physical core per CCD (CCD k = cores 8k..8k+7); slots 0..3 -> CCD 0..3.
# CORES_LIST lets a second, concurrent instance of this driver take a disjoint
# CCD group (e.g. "64 72 80 88" = CCD 8..11) so a long row can overlap the rest
# of the sweep while every cell still owns one core on its own CCD.
read -r -a CORES <<< "${CORES_LIST:-0 8 16 24}"

# Concurrent pinned cells on OTHER CCDs are part of the campaign design, so the
# sibling check is off by default; set SIBLING_CHECK=1 to require exclusivity.
sibling_count() {
  [ "${SIBLING_CHECK:-0}" = "1" ] || { echo 0; return; }
  pgrep -u "$ME" -f 'python -m pplbench|harness\.jl' 2>/dev/null | wc -l
}

wait_for_quiet() {
  local waited=0 l e s
  while :; do
    l="$(ambient_load1 | cut -d. -f1)"; e="$(external_cpu)"; s="$(sibling_count)"
    if [ "$l" -lt "$LOAD_GATE" ] && [ "$e" -lt "${START_EXT_CPU:-$EXT_CPU_LIMIT}" ] && [ "$s" -eq 0 ] && ! /mnt4/sbrantq/probprog/oopsla/tools/bazel_building.sh; then
      [ "$waited" -gt 0 ] && note "  (waited ${waited}s for quiet; load1=$l ext=${e}%)"
      return 0
    fi
    [ $((waited % 900)) -eq 0 ] && note "  waiting ${waited}s: load1=$l ext=${e}% sibling=$s"
    sleep 30; waited=$((waited + 30))
  done
}

declare -A PIDS=()
declare -A CELLSTART=()
ROW_T0=0
WATCHDOG_PID=""

kill_tree() {
  local p="$1" k
  for k in $(ps --ppid "$p" -o pid= 2>/dev/null); do kill_tree "$k"; done
  kill -TERM "$p" 2>/dev/null
}

# A cell measured through an external burst is not merely flagged, it is wrong:
# on this machine a pinned cell keeps 99% of its core under external load 200
# (verified with /usr/bin/time) and still runs 2x slower, because the other
# 145 busy cores take the memory bandwidth and the boost headroom, and the
# slowdown is not uniform across systems. So a row whose conditions go bad is
# ABORTED as soon as that is clear, rather than run to completion and thrown
# away: the retry then starts in the next quiet window instead of hours later.
start_watchdog() {
  local pids="$1"
  ( bad=0
    while :; do
      sleep 5
      alive=0
      for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done
      [ "$alive" -eq 0 ] && exit 0
      a="$(ambient_load1)"; e="$(external_cpu)"
      if awk -v a="$a" -v m="$LOAD_MAX" -v e="$e" -v l="$EXT_CPU_LIMIT" \
             'BEGIN{exit !(a>m || e>=l)}'; then
        bad=$((bad + 1))
      else
        bad=0
      fi
      if [ "$bad" -ge "${ABORT_SAMPLES:-24}" ]; then
        echo "[$(date '+%F %T')]   WATCHDOG: ambient load1=$a external CPU=${e}% for $((bad * 5))s; aborting the row for retry" | tee -a "$DRIVER_LOG"
        for p in $pids; do kill_tree "$p"; done
        exit 1
      fi
    done ) &
  WATCHDOG_PID=$!
}

spawn() {   # spawn <label> <slot> <timeout> <run_pplbench args...>
  local label="$1" slot="$2" tmo="$3"; shift 3
  local core="${CORES[$slot]}"
  # ONLY_PPL (2026-09-22): run only the cells of one system, e.g. ONLY_PPL=impulse.
  if [ -n "${ONLY_PPL:-}" ] && [[ " $* " != *" --ppls $ONLY_PPL "* ]]; then return 0; fi
  note "SPAWN $label (core $core): run_pplbench.sh $*"
  CELLSTART[$label]=$(date +%s)
  # /usr/bin/time records this cell's OWN cpu seconds; cpu/wall on a
  # single-core-pinned cell is the direct test of whether its core was stolen.
  PPLBENCH_TASKSET_CPUS="$core" /usr/bin/time \
      -f "cell_elapsed_s=%e cell_user_s=%U cell_sys_s=%S cell_cpu_pct=%P" \
      -o "$LOGDIR/cells/${label}.time.txt" \
      timeout "$tmo" ./run_pplbench.sh "$@" \
      > "$LOGDIR/${label}.log" 2>&1 &
  PIDS[$!]="$label"
  sleep 3   # pplbench output dirs are second-resolution timestamps
}

ROW_FAILED=0
wait_row() {   # wait_row <rowid>
  local pid rc label t0 t1 lmax emax n
  ROW_FAILED=0
  for pid in "${!PIDS[@]}"; do
    label="${PIDS[$pid]}"
    if wait "$pid"; then rc=0; else rc=$?; fi
    t1=$(date +%s); t0="${CELLSTART[$label]}"
    # Contamination statistics over the cell window. The INSTANTANEOUS max of
    # external CPU is recorded for every cell, but the rerun decision uses the
    # SUSTAINED level: the largest 5-minute (60-sample) rolling mean of external
    # CPU. Ambient external load on this shared box sits at 500-900% today, so a
    # single 5 s sample above 1000% is a transient that cannot move a cell
    # pinned to its own idle core, while a 5-minute mean above 1000% is a real
    # competing job. Both numbers, plus the fraction of samples at or above the
    # limit, go into the per-cell gate file.
    read -r lmax emax amax esus asus efrac n <<< "$(awk -F, -v a="$t0" -v b="$t1" \
        'NR>1 && $1>=a && $1<=b {
             if ($2+0>l) l=$2+0; if ($3+0>e) e=$3+0; if ($5+0>m) m=$5+0
             c++; v[c]=$3+0; w[c]=$5+0
         }
         END {
             W = (c < 60 ? c : 60); beste = 0; besta = 0
             if (c > 0) {
                 re = 0; ra = 0
                 for (i = 1; i <= c; i++) {
                     re += v[i]; ra += w[i]
                     if (i > W) { re -= v[i-W]; ra -= w[i-W] }
                     if (i >= W) {
                         if (re / W > beste) beste = re / W
                         if (ra / W > besta) besta = ra / W
                     }
                 }
                 over = 0
                 for (i = 1; i <= c; i++) if (v[i] >= LIM) over++
             }
             printf "%.2f %.0f %.2f %.0f %.2f %.4f %d", l+0, e+0, m+0, beste+0,
                    besta+0, (c ? over / c : 0), c+0
         }' LIM="$EXT_CPU_LIMIT" "$LOAD_CSV")"
    {
      echo "cell=$label rc=$rc t_start=$t0 t_end=$t1 dur_s=$((t1-t0))"
      echo "load1_max=$lmax ambient_load1_max=$amax extcpu_max=$emax"
      echo "extcpu_sustained_max_5min=$esus ambient_load1_sustained_max_5min=$asus"
      echo "extcpu_frac_samples_over_limit=$efrac n_samples=$n"
      cat "$LOGDIR/cells/${label}.time.txt" 2>/dev/null
      echo "gates: LOAD_GATE=$LOAD_GATE LOAD_MAX=$LOAD_MAX EXT_CPU_LIMIT=$EXT_CPU_LIMIT"
      awk -F, -v a="$t0" -v b="$t1" 'NR>1 && $1>=a && $1<=b' "$LOAD_CSV" \
        | awk -F, '{s+=$2; e+=$3; n++} END {if(n) printf "load1_mean=%.2f extcpu_mean=%.0f\n", s/n, e/n}'
    } > "$LOGDIR/cells/${label}.gate.txt"
    if [ "$rc" -ne 0 ]; then
      note "END   $label: FAILED (rc=$rc) load1_max=$lmax extcpu_max=$emax"
      ROW_FAILED=1
    elif [ "${STRICT:-0}" = "1" ] && [ "$emax" -ge "$EXT_CPU_LIMIT" ]; then
      note "END   $label: CONTAMINATED (strict: extcpu instantaneous max=$emax >= $EXT_CPU_LIMIT)"
      ROW_FAILED=1
    elif awk -v l="$asus" -v m="$LOAD_MAX" 'BEGIN{exit !(l>m)}'; then
      note "END   $label: CONTAMINATED (ambient load1 sustained 5-min mean=$asus > $LOAD_MAX; instantaneous max=$amax, raw load1 max=$lmax, extcpu_max=$emax)"
      ROW_FAILED=1
    elif awk -v e="$esus" -v m="$EXT_CPU_LIMIT" 'BEGIN{exit !(e>=m)}'; then
      note "END   $label: CONTAMINATED (extcpu sustained 5-min mean=$esus >= $EXT_CPU_LIMIT; instantaneous max=$emax, frac over limit=$efrac, load1_max=$lmax, ambient=$amax)"
      ROW_FAILED=1
    else
      note "END   $label: OK load1_max=$lmax ambient_max=$amax ambient_sustained=$asus extcpu_max=$emax extcpu_sustained=$esus frac_over=$efrac dur=$((t1-t0))s"
    fi
  done
  # the watchdog exits by itself once every cell is gone; reap it either way
  if [ -n "$WATCHDOG_PID" ]; then
    kill "$WATCHDOG_PID" 2>/dev/null
    wait "$WATCHDOG_PID" 2>/dev/null
    WATCHDOG_PID=""
  fi
  PIDS=()
  local now; now="$(so_md5)"
  [ "$now" != "$BASE_MD5" ] && { note "!! BINARY CHANGED during $1: $BASE_MD5 -> $now"; ROW_FAILED=1; }
  return 0
}

# ------------------------------------------------------------------ rows
run_row() {   # run_row <rowid>
  local row="$1" attempt=1
  while :; do
    wait_for_quiet
    note "-- ROW $row (attempt $attempt) load1=$(load1) ambient=$(ambient_load1) ext=$(external_cpu)%"
    local sfx=""; [ "$attempt" -gt 1 ] && sfx="_r${attempt}"
    case "$row" in
      logistic)
        spawn "nuts_logistic_regression_numpyro${sfx}" 0 6h --threads 1 --ppls numpyro standard/logistic_regression_matched_init
        spawn "nuts_logistic_regression_impulse${sfx}" 1 6h --threads 1 --ppls impulse standard/logistic_regression_matched_init
        spawn "nuts_logistic_regression_turing${sfx}"  2 6h --threads 1 --ppls turing  standard/logistic_regression_matched_init
        spawn "nuts_logistic_regression_stan${sfx}"    3 6h --threads 1 --ppls stan    standard/logistic_regression_matched_init ;;
      robust)
        spawn "nuts_robust_regression_numpyro${sfx}" 0 6h --threads 1 --ppls numpyro standard/robust_regression_matched_init
        spawn "nuts_robust_regression_impulse${sfx}" 1 6h --threads 1 --ppls impulse standard/robust_regression_matched_init
        spawn "nuts_robust_regression_turing${sfx}"  2 6h --threads 1 --ppls turing  standard/robust_regression_matched_init
        spawn "nuts_robust_regression_stan${sfx}"    3 6h --threads 1 --ppls stan    standard/robust_regression_matched_init ;;
      n_schools)
        spawn "nuts_n_schools_numpyro${sfx}" 0 6h --threads 1 --ppls numpyro standard/n_schools_matched_init
        spawn "nuts_n_schools_impulse${sfx}" 1 6h --threads 1 --ppls impulse standard/n_schools_matched_init
        spawn "nuts_n_schools_turing${sfx}"  2 6h --threads 1 --ppls turing  standard/n_schools_matched_init
        spawn "nuts_n_schools_stan${sfx}"    3 6h --threads 1 --ppls stan    standard/n_schools_matched_init ;;
      annotation)
        # Impulse from the PLAIN config, baselines from matched_init (as in the
        # 2026-08-24 sweep and the figure's provenance).
        spawn "nuts_crowd_sourced_annotation_numpyro${sfx}" 0 8h --threads 1 --ppls numpyro standard/crowd_sourced_annotation_matched_init
        spawn "nuts_crowd_sourced_annotation_impulse${sfx}" 1 8h --threads 1 --ppls impulse standard/crowd_sourced_annotation
        spawn "nuts_crowd_sourced_annotation_turing${sfx}"  2 8h --threads 1 --ppls turing  standard/crowd_sourced_annotation_matched_init
        spawn "nuts_crowd_sourced_annotation_stan${sfx}"    3 8h --threads 1 --ppls stan    standard/crowd_sourced_annotation_matched_init ;;
      noisy_or)
        spawn "nuts_noisy_or_topic_numpyro${sfx}" 0 8h --threads 1 --ppls numpyro standard/noisy_or_topic_matched_init
        spawn "nuts_noisy_or_topic_impulse${sfx}" 1 8h --threads 1 --ppls impulse standard/noisy_or_topic_matched_init
        spawn "nuts_noisy_or_topic_turing${sfx}"  2 8h --threads 1 --ppls turing  standard/noisy_or_topic_matched_init
        spawn "nuts_noisy_or_topic_stan${sfx}"    3 8h --threads 1 --ppls stan    standard/noisy_or_topic_matched_init ;;
      noisy_or_w1000)
        spawn "nuts_noisy_or_topic_w1000_numpyro${sfx}" 0 10h --threads 1 --ppls numpyro standard/noisy_or_topic_w1000_matched_init
        spawn "nuts_noisy_or_topic_w1000_impulse${sfx}" 1 10h --threads 1 --ppls impulse standard/noisy_or_topic_w1000_matched_init
        spawn "nuts_noisy_or_topic_w1000_turing${sfx}"  2 10h --threads 1 --ppls turing  standard/noisy_or_topic_w1000_matched_init
        spawn "nuts_noisy_or_topic_w1000_stan${sfx}"    3 10h --threads 1 --ppls stan    standard/noisy_or_topic_w1000_matched_init ;;
      svi_logistic)
        spawn "svi_logistic_numpyro${sfx}" 0 5h --threads 1 --ppls numpyro svi/logistic_regression_plateau_budget
        spawn "svi_logistic_impulse${sfx}" 1 5h --threads 1 --ppls impulse svi/logistic_regression_plateau_budget
        spawn "svi_logistic_turing${sfx}"  2 5h --threads 1 --ppls turing  svi/logistic_regression_plateau_budget
        spawn "svi_logistic_stan${sfx}"    3 5h --threads 1 --ppls stan    svi/logistic_regression_matched_init ;;
      svi_robust)
        spawn "svi_robust_regression_numpyro${sfx}" 0 4h --threads 1 --ppls numpyro svi/robust_regression
        spawn "svi_robust_regression_impulse${sfx}" 1 4h --threads 1 --ppls impulse svi/robust_regression
        spawn "svi_robust_regression_turing${sfx}"  2 4h --threads 1 --ppls turing  svi/robust_regression
        spawn "svi_robust_regression_stan${sfx}"    3 4h --threads 1 --ppls stan    svi/robust_regression ;;
      svi_n_schools)
        spawn "svi_n_schools_numpyro${sfx}" 0 4h --threads 1 --ppls numpyro svi/n_schools
        spawn "svi_n_schools_impulse${sfx}" 1 4h --threads 1 --ppls impulse svi/n_schools
        spawn "svi_n_schools_turing${sfx}"  2 4h --threads 1 --ppls turing  svi/n_schools
        spawn "svi_n_schools_stan${sfx}"    3 4h --threads 1 --ppls stan    svi/n_schools ;;
      *) note "!! unknown row $row"; return 1 ;;
    esac
    start_watchdog "${!PIDS[*]}"
    wait_row "$row"
    if [ "$ROW_FAILED" -eq 0 ]; then
      note "ROW $row: CLEAN (attempt $attempt)"
      return 0
    fi
    if [ "$attempt" -ge "$MAX_ROW_RETRIES" ]; then
      note "!! ROW $row: giving up after $attempt attempts (cells kept, marked contaminated)"
      return 1
    fi
    attempt=$((attempt + 1))
    note "ROW $row: rerunning (attempt $attempt)"
  done
}

ROWS="${ROWS:-logistic robust n_schools annotation noisy_or noisy_or_w1000 svi_logistic svi_robust svi_n_schools}"
note "sweep start; rows: $ROWS; logdir $LOGDIR"
for r in $ROWS; do run_row "$r"; done
note "sweep done; md5 at end = $(so_md5)"
