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

BENCH="/mnt/sbrantq/probprog/oopsla/Reactant.jl/benchmark/probprog"
SO="/mnt/sbrantq/probprog/oopsla/Reactant.jl/deps/ReactantExtra/bazel-bin/libReactantExtra.so"
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
EXT_CPU_LIMIT="${FROZEN_EXT_CPU_LIMIT:-1000}"
LOAD_GATE="${FROZEN_LOAD_GATE:-12}"
MAX_RETRIES="${FROZEN_MAX_RETRIES:-3}"

# model_key : julia_model_path : step_size : max_tree_depth
CELLS="${FROZEN_CELLS:-
car_spatial_scotland:sicm/car_spatial:0.05:10
car_spatial_nyc:sicm/car_spatial:0.05:10
pgls:sicm/pgls:0.05:10
hierarchical_mvn:sicm/hierarchical_mvn:0.01:10
lmm_radon:sicm/linear_mixed_model:0.05:10
sar_columbus:sicm/sar_lag:0.05:10
diamonds:sicm/diamonds_regression:0.05:10
}"

external_cpu () {
  ps -eo user,pcpu --no-headers 2>/dev/null \
    | awk -v me="$ME" '$1 != me { s += $2 } END { printf "%.0f", s+0 }'
}
sibling_count () { pgrep -u "$ME" -f "harness.jl|python -m pplbench" 2>/dev/null | wc -l; }

wait_for_quiet () {
  local waited=0 busy load1 ext
  while :; do
    busy="$(sibling_count)"; load1="$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)"
    ext="$(external_cpu)"
    if [ "$busy" -eq 0 ] && [ "$load1" -lt "$LOAD_GATE" ] && [ "$ext" -lt "$EXT_CPU_LIMIT" ]; then
      [ "$waited" -gt 0 ] && echo "    (waited ${waited}s; load1=$load1 ext=${ext}%)" >> "$META"
      return 0
    fi
    [ $((waited % 600)) -eq 0 ] && \
      echo "    waiting ${waited}s: sibling=$busy load1=$load1 ext=${ext}% $(date -Is)" >> "$META"
    sleep 30; waited=$((waited + 30))
  done
}

{
  echo "batch tag: $TAG"
  echo "protocol: frozen adapted state, adaptation OFF both arms, matched seeds"
  echo "num_samples=$NUM_SAMPLES num_exec=$NUM_EXEC warmup=$WARMUP"
  echo "ext_cpu_limit=${EXT_CPU_LIMIT}% load_gate=$LOAD_GATE max_retries=$MAX_RETRIES"
  echo "start: $(date -Is)"
  echo "md5 at batch start: $(md5sum "$SO" | cut -d' ' -f1)"
} > "$META"

cd "$BENCH" || exit 1

for spec in $CELLS; do
  key="${spec%%:*}"; rest="${spec#*:}"
  model="${rest%%:*}"; rest="${rest#*:}"
  step="${rest%%:*}"; depth="${rest##*:}"
  data="$HERE/data/${key}.json"
  state="$STATES/${key}.json"
  [ -f "$data" ] || { echo "!! no data for $key" >> "$META"; continue; }

  # ---- warmup once (adaptation ON) -> adapted state --------------------
  if [ ! -f "$state" ] || [ "${FROZEN_RESAVE:-0}" = "1" ]; then
    wait_for_quiet
    echo "--- warmup $key  $(date -Is)  ext=$(external_cpu)%" >> "$META"
    CUDA_VISIBLE_DEVICES="" JULIA_CI=true timeout 21600 \
      julia -t auto --project=. harness.jl --model "$model" --data "$data" \
        --output /tmp/sbrantq-frozen-warm-$key.json \
        --num-warmup "$WARMUP" --num-samples 1 --seed 1396133 \
        --step-size "$step" --max-tree-depth "$depth" \
        --save-state "$state" > "$LOGS/warmup_${key}.log" 2>&1
    grep -E "adapted step_size|wrote" "$LOGS/warmup_${key}.log" >> "$META"
  else
    echo "--- warmup $key: reusing existing $state" >> "$META"
  fi
  [ -f "$state" ] || { echo "!! warmup failed for $key" >> "$META"; continue; }

  # ---- measured executions, both arms ---------------------------------
  for arm in sicm nosicm; do
    flag=""; [ "$arm" = "nosicm" ] && flag="--disable-sicm"
    for e in 1; do
      tag="${key}_${arm}"
      attempt=1
      while :; do
        wait_for_quiet
        extfile="$LOGS/.ext_$tag"; echo 0 > "$extfile"
        ( while :; do
            c="$(external_cpu)"; m="$(cat "$extfile" 2>/dev/null || echo 0)"
            [ "$c" -gt "$m" ] && echo "$c" > "$extfile"
            sleep 5
          done ) & sampler=$!
        echo "--- cell $tag (attempt $attempt) ext_before=$(external_cpu)% load=$(cut -d' ' -f1 /proc/loadavg) $(date -Is)" >> "$META"
        CUDA_VISIBLE_DEVICES="" JULIA_CI=true timeout 21600 \
          julia -t auto --project=. harness.jl --model "$model" --data "$data" \
            --output "$LOGS/${tag}.out.json" \
            --num-warmup 0 --num-samples "$NUM_SAMPLES" --seed 1396133 \
            --step-size "$step" --max-tree-depth "$depth" \
            --load-state "$state" --repeat $((NUM_EXEC + 1)) $flag \
            > "$LOGS/${tag}.log" 2>&1
        rc=$?
        kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
        ext_max="$(cat "$extfile" 2>/dev/null || echo 0)"
        md5_now="$(md5sum "$SO" | cut -d' ' -f1)"
        if [ "$rc" -eq 0 ] && [ "$ext_max" -lt "$EXT_CPU_LIMIT" ]; then
          rt="$(grep -oE 'repeats \(steady state, s\): .*' "$LOGS/${tag}.log" | tail -1)"
          echo "    OK $tag ${rt} ext_max=${ext_max}% md5=$md5_now" >> "$META"
          break
        fi
        echo "!! CONTAMINATED/FAILED $tag (rc=$rc ext_max=${ext_max}% limit=${EXT_CPU_LIMIT}%)" >> "$META"
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
