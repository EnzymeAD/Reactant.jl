#!/bin/bash
# The SlicStan comparison (figure 12 protocol) on the section-2 phylogenetic
# Poisson regression, one gated cell per program variant.
#
# Protocol of run_slicstan_per_variant.sh unchanged except the model: 500
# iterations / 200 warmup / 4 trials, per-variant seed 20260822, step 0.05,
# depth 10, adaptation on, single thread pinned by PPLBENCH_TASKSET_CPUS, every
# cell through refreeze_cell.sh with the same gates. Configs:
# pplbench_configs/related_work/phylo_pois_ladder_4trials_<variant>.json.
#
# Differences from the GP driver, both about scheduling, not measurement:
#   * ONE core, one cell at a time (CORES has one entry by default);
#   * before popping a cell the worker waits while $PAUSE exists (so an
#     uncapped measurement can have the machine to itself) and checks that the
#     binary md5 equals EXPECT_MD5 (the measurement binary); on a mismatch it
#     stops instead of measuring on another binary.
set -uo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
cd "$B" || exit 1
LOG="${1:?usage: run_slicstan_phylo_per_variant.sh <logdir>}"
mkdir -p "$LOG/cells"
SO="$B/../../deps/ReactantExtra/bazel-bin/libReactantExtra.so"
EXPECT_MD5="${EXPECT_MD5:?set EXPECT_MD5 to the measurement binary md5}"
PAUSE="${PAUSE:-$LOG/.pause}"

G="LOAD_GATE=${LOAD_GATE:-20} START_EXT_CPU=${START_EXT_CPU:-1000} LOAD_MAX=${LOAD_MAX:-24} \
EXT_CPU_LIMIT=${EXT_CPU_LIMIT:-1000} ABORT_SAMPLES=${ABORT_SAMPLES:-24} MAX_RETRIES=${MAX_RETRIES:-200} STRICT=${STRICT:-1}"

QUEUE="$LOG/.variant_queue"
LOCK="$LOG/.variant_queue.lock"
rm -rf "$LOCK"
if [ ! -f "$QUEUE" ]; then
cat > "$QUEUE" <<'EOQ'
phylo_impulse_sicm impulse phylo_pois_ladder_4trials_impulse_sicm 2h
phylo_stan_expert stan phylo_pois_ladder_4trials_stan_expert 3h
phylo_stan_naive stan phylo_pois_ladder_4trials_stan_naive 8h
phylo_stan_slicstan stan phylo_pois_ladder_4trials_stan_slicstan 8h
phylo_impulse_no_preopt impulse phylo_pois_ladder_4trials_impulse_no_preopt 8h
phylo_impulse_std_passes impulse phylo_pois_ladder_4trials_impulse_std_passes 8h
EOQ
fi
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
  local core="$1" line label ppl cfg tmo rc md5
  while :; do
    while [ -e "$PAUSE" ]; do sleep 30; done
    md5="$(md5sum "$SO" | cut -d' ' -f1)"
    if [ "$md5" != "$EXPECT_MD5" ]; then
      echo "[$(date '+%F %T')] core $core STOP: binary md5 $md5 != expected $EXPECT_MD5"
      break
    fi
    line="$(pop)"
    [ -z "$line" ] && break
    set -- $line
    label="$1"; ppl="$2"; cfg="$3"; tmo="$4"
    echo "[$(date '+%F %T')] core $core START $label ($cfg, timeout $tmo) md5=$md5"
    env $G ./refreeze_cell.sh "$label" "$LOG" "$tmo" none -- \
      "PPLBENCH_TASKSET_CPUS=$core ./run_pplbench.sh --threads 1 --ppls $ppl related_work/$cfg" \
      > "$LOG/$label.driver.log" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] core $core DONE  $label rc=$rc verdict=$(grep -h '^verdict=' "$LOG/cells/$label.gate.txt" 2>/dev/null | tail -1) md5_after=$(md5sum "$SO" | cut -d' ' -f1)"
  done
}

CORES="${CORES:-48}"
echo "[$(date '+%F %T')] phylo SlicStan per-variant pass: cores $CORES, gates: $G, expected md5 $EXPECT_MD5"
for c in $CORES; do worker "$c" & done
wait
echo "[$(date '+%F %T')] phylo SlicStan pass done"
grep -H '^verdict=' "$LOG"/cells/phylo_*.gate.txt 2>/dev/null
