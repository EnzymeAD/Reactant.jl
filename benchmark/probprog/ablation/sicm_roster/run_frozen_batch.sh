#!/bin/bash
# Frozen-adapted-state batch for the SICM ablation roster.
#
# Protocol (matches the pre-lowering session's, adapted to these models):
#   per model: warm up ONCE with adaptation ON, save the adapted state
#              (step size, dense inverse mass, post-warmup draw);
#   then BOTH arms measure with adaptation OFF from that shared state with
#   matched seeds, so the arms take the same trajectory and the wall-clock
#   ratio is a compiled per-step-cost ratio rather than a trajectory-length
#   artifact (which is what sank the DotScale row on the naive protocol).
#
# NUM_SAMPLES draws; NUM_EXEC timed executions of the SAME compiled thunk
# IN ONE PROCESS (--repeat), discarding the first call. Separate processes would
# make every execution a first call, whose warm-up dominates a short cell and
# dilutes the arm ratio (measured on Scotland: 1.79x across processes vs 4.6x
# steady state). Median and geomean reported
# by summarize_frozen.py.
#
# Contamination gate ADOPTED VERBATIM in spirit from
# ablation/pre_lowering/run_prelowering_batch_v5.sh: raw loadavg is a bad test
# because a heavy cell self-drives it, so we sample the summed %CPU of processes
# NOT owned by this user every 5 s for the duration of every cell and retry any
# cell whose external CPU ever exceeds EXT_CPU_LIMIT. That gate caught a 16x
# measurement error in their campaign that per-cell before/after load sampling
# missed.
set -uo pipefail

BENCH="/mnt4/sbrantq/probprog/oopsla/Reactant.jl/benchmark/probprog"
SO="/mnt4/sbrantq/probprog/oopsla/Reactant.jl/deps/ReactantExtra/bazel-bin/libReactantExtra.so"
HERE="$BENCH/ablation/sicm_roster"
TAG="${1:-frozen1}"
LOGS="$HERE/logs_frozen/$TAG"
STATES="$HERE/adapted_states"
META="$LOGS/batch_meta.txt"
mkdir -p "$LOGS" "$STATES"

ME="$(id -un)"
NUM_SAMPLES="${FROZEN_NUM_SAMPLES:-20}"
NUM_EXEC="${FROZEN_NUM_EXEC:-5}"
WARMUP="${FROZEN_WARMUP:-200}"
EXT_CPU_LIMIT="${FROZEN_EXT_CPU_LIMIT:-1000}"   # during-cell contamination limit
START_EXT_CPU="${FROZEN_START_EXT_CPU:-300}"    # stricter gate to START a cell
LOAD_GATE="${FROZEN_LOAD_GATE:-8}"              # ambient load1 gate (uncapped regime)
MAX_RETRIES="${FROZEN_MAX_RETRIES:-40}"
ABORT_SAMPLES="${FROZEN_ABORT_SAMPLES:-24}"   # 24 x 5 s = 2 min of bad conditions aborts the cell
# Seed for BOTH the warmup that mints the adapted state and the measured
# executions. Exposed so a row whose two arms decorrelate inside the measured
# window (a ULP flip flipping a discrete sampler decision) can be retried at a
# different seed until the matched-trajectory gate passes.
SEED="${FROZEN_SEED:-1396133}"
# States are per (model, seed): a state minted at one seed must not be measured
# at another.
STATE_SUFFIX=""
[ "$SEED" != "1396133" ] && STATE_SUFFIX="_seed$SEED"

# model_key : julia_model_path : step_size : max_tree_depth
CELLS="${FROZEN_CELLS:-
car_spatial_scotland:sicm/car_spatial:0.05:10
car_spatial_nyc:sicm/car_spatial:0.05:10
pgls:sicm/pgls:0.05:10
hierarchical_mvn:sicm/hierarchical_mvn:0.01:10
lmm_radon:sicm/linear_mixed_model:0.05:10
sar_columbus:sicm/sar_lag:0.05:10
diamonds:sicm/diamonds_regression:0.05:10
phylo_pois:motivating/phylo_pois:0.1:10
}"

external_cpu () {
  ps -eo user,pcpu --no-headers 2>/dev/null \
    | awk -v me="$ME" '$1 != me { s += $2 } END { printf "%.0f", s+0 }'
}
kill_tree () {
  local p="$1" k
  for k in $(ps --ppid "$p" -o pid= 2>/dev/null); do kill_tree "$k"; done
  kill -TERM "$p" 2>/dev/null
}
# load1 is a ONE-MINUTE exponential average: after a burst that drove it to 250
# it needs four to five minutes of idle to fall below 8, which is longer than
# this machine's quiet windows, so a raw-load1 gate would never open even on an
# idle machine. procs_running from /proc/stat is the same quantity (how many
# tasks are runnable right now) without the lag, and the external-CPU sum is
# already instantaneous. The gate is therefore: external CPU below the limit
# AND fewer than LOAD_GATE runnable tasks, both holding for QUIET_HOLD seconds.
procs_running () { awk '/^procs_running/ {print $2}' /proc/stat; }
sibling_count () { pgrep -u "$ME" -f "harness.jl|python -m pplbench" 2>/dev/null | wc -l; }

QUIET_HOLD="${FROZEN_QUIET_HOLD:-30}"
wait_for_quiet () {
  local waited=0 busy load1 ext run held=0
  while :; do
    busy="$(sibling_count)"; load1="$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)"
    ext="$(external_cpu)"; run="$(procs_running)"
    if [ "$busy" -eq 0 ] && [ "$run" -lt "$LOAD_GATE" ] && [ "$ext" -lt "$START_EXT_CPU" ] && ! /mnt4/sbrantq/probprog/oopsla/tools/bazel_building.sh; then
      held=$((held + 5))
      if [ "$held" -ge "$QUIET_HOLD" ]; then
        [ "$waited" -gt 0 ] && echo "    (waited ${waited}s; procs_running=$run load1=$load1 ext=${ext}%)" >> "$META"
        return 0
      fi
      sleep 5; waited=$((waited + 5)); continue
    fi
    held=0
    [ $((waited % 600)) -lt 30 ] && \
      echo "    waiting ${waited}s: sibling=$busy procs_running=$run load1=$load1 ext=${ext}% $(date -Is)" >> "$META"
    sleep 30; waited=$((waited + 30))
  done
}

{
  echo "batch tag: $TAG"
  echo "protocol: frozen adapted state, adaptation OFF both arms, matched seeds"
  echo "num_samples=$NUM_SAMPLES num_exec=$NUM_EXEC warmup=$WARMUP seed=$SEED"
  echo "ext_cpu_limit=${EXT_CPU_LIMIT}% start_ext_cpu=${START_EXT_CPU}% load_gate=$LOAD_GATE max_retries=$MAX_RETRIES"
  echo "start: $(date -Is)"
  echo "md5 at batch start: $(md5sum "$SO" | cut -d' ' -f1)"
} > "$META"

cd "$BENCH" || exit 1

for spec in $CELLS; do
  key="${spec%%:*}"; rest="${spec#*:}"
  model="${rest%%:*}"; rest="${rest#*:}"
  step="${rest%%:*}"; depth="${rest##*:}"
  data="$HERE/data/${key}.json"
  state="$STATES/${key}${STATE_SUFFIX}.json"
  [ -f "$data" ] || { echo "!! no data for $key" >> "$META"; continue; }

  # ---- warmup once (adaptation ON) -> adapted state --------------------
  if [ ! -f "$state" ] || [ "${FROZEN_RESAVE:-0}" = "1" ]; then
    wait_for_quiet
    echo "--- warmup $key  $(date -Is)  ext=$(external_cpu)%" >> "$META"
    CUDA_VISIBLE_DEVICES="" JULIA_CI=true timeout 21600 \
      julia -t auto --project=. harness.jl --model "$model" --data "$data" \
        --output /tmp/sbrantq-frozen-warm-$key.json \
        --num-warmup "$WARMUP" --num-samples 1 --seed "$SEED" \
        --step-size "$step" --max-tree-depth "$depth" \
        --save-state "$state" > "$LOGS/warmup_${key}.log" 2>&1
    grep -E "adapted step_size|wrote" "$LOGS/warmup_${key}.log" >> "$META"
  else
    echo "--- warmup $key: reusing existing $state" >> "$META"
  fi
  [ -f "$state" ] || { echo "!! warmup failed for $key" >> "$META"; continue; }

  # ---- measured executions, both arms ---------------------------------
  # FROZEN_ARMS (additive, default both): re-measure one configuration of a
  # row whose other configuration is already accepted in this tag.
  for arm in ${FROZEN_ARMS:-sicm nosicm}; do
    flag=""; [ "$arm" = "nosicm" ] && flag="--disable-sicm"
    for e in 1; do
      tag="${key}${STATE_SUFFIX}_${arm}"
      attempt=1
      while :; do
        wait_for_quiet
        extfile="$LOGS/.ext_$tag"; echo 0 > "$extfile"
        csv="$LOGS/${tag}.samples.csv"; echo "epoch,load1,extcpu" > "$csv"
        ( while :; do
            c="$(external_cpu)"; m="$(cat "$extfile" 2>/dev/null || echo 0)"
            [ "$c" -gt "$m" ] && echo "$c" > "$extfile"
            echo "$(date +%s),$(cut -d' ' -f1 /proc/loadavg),$c" >> "$csv"
            sleep 5
          done ) & sampler=$!
        echo "--- cell $tag (attempt $attempt) ext_before=$(external_cpu)% load=$(cut -d' ' -f1 /proc/loadavg) $(date -Is)" >> "$META"
        CUDA_VISIBLE_DEVICES="" JULIA_CI=true timeout 21600 \
          julia -t auto --project=. harness.jl --model "$model" --data "$data" \
            --output "$LOGS/${tag}.out.json" \
            --num-warmup 0 --num-samples "$NUM_SAMPLES" --seed "$SEED" \
            --step-size "$step" --max-tree-depth "$depth" \
            --load-state "$state" --repeat $((NUM_EXEC + 1)) $flag \
            > "$LOGS/${tag}.log" 2>&1 &
        cellpid=$!
        ( bad=0
          while kill -0 "$cellpid" 2>/dev/null; do
            sleep 5
            e="$(external_cpu)"; l1="$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)"
            # External CPU ONLY. Raw load1 is not a contamination test in the
            # uncapped regime: a heavy ablation cell drives load1 to 30-70 by
            # itself through multithreaded LAPACK. Gating on it aborted healthy
            # cells (heat_s3_B was killed four times at external CPU 597-689%,
            # well inside the 1000% limit, because its own load1 reached 38-41).
            if [ "$e" -ge "$EXT_CPU_LIMIT" ]; then
              bad=$((bad + 1))
            else
              bad=0
            fi
            if [ "$bad" -ge "$ABORT_SAMPLES" ]; then
              echo "    WATCHDOG $tag: external CPU=${e}% load1=$l1 for $((bad * 5))s; aborting attempt $attempt" >> "$META"
              kill_tree "$cellpid"
              exit 1
            fi
          done ) &
        wd=$!
        wait "$cellpid"
        rc=$?
        kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null
        kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
        ext_max="$(cat "$extfile" 2>/dev/null || echo 0)"
        load_max="$(awk -F, 'NR>1 && $2+0>m {m=$2+0} END {printf "%.2f", m+0}' "$csv")"
        md5_now="$(md5sum "$SO" | cut -d' ' -f1)"
        if [ "$rc" -eq 0 ] && [ "$ext_max" -lt "$EXT_CPU_LIMIT" ]; then
          rt="$(grep -oE 'repeats \(steady state, s\): .*' "$LOGS/${tag}.log" | tail -1)"
          echo "    OK $tag ${rt} ext_max=${ext_max}% load1_max=${load_max} md5=$md5_now" >> "$META"
          break
        fi
        echo "!! CONTAMINATED/FAILED $tag (rc=$rc ext_max=${ext_max}% limit=${EXT_CPU_LIMIT}% load1_max=${load_max})" >> "$META"
        attempt=$((attempt + 1))
        if [ "$attempt" -gt "$MAX_RETRIES" ]; then
          echo "!! GIVING UP on $tag after $MAX_RETRIES attempts" >> "$META"; break
        fi
        sleep 60
      done
    done
  done
done

{
  echo "end: $(date -Is)"
  echo "md5 at batch end: $(md5sum "$SO" | cut -d' ' -f1)"
} >> "$META"
cat "$META"
