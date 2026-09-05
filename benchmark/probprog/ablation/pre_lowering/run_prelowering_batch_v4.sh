#!/usr/bin/env bash
# Frozen-adapted-state batch for the four figure rows of tab:prelowering
# (2026-08-25). Supersedes v3, which measured from a naive fixed step size.
#
# Per model:
#   PHASE 1 (not timed): warmup with adaptation ON (step size + mass matrix) in
#     the Pre-Opt arm, PRELOWER_WARMUP iterations. Writes adapted_states/<m>.json
#     holding the adapted step size, adapted inverse mass and post-warmup
#     position, with provenance (which arm, seed, warmup length).
#   PHASE 2 (timed): BOTH arms with adaptation OFF, initialised from that SAME
#     state. Identical step size, identical inverse mass, identical starting
#     position, matched seeds, so the arms walk bit-identical trajectories and
#     the identical-accepts gate is still the arms-equivalence check.
#
# Prerequisite: verify_state_injection.jl must have passed on a cheap model.
#
# Machine discipline unchanged from v3: load1 < 8 AND no pplbench/harness.jl
# before EVERY cell, with NO proceed-anyway ceiling; per-cell binary md5; a flag
# if a sibling job appears mid-cell.
#
# Run from the Reactant.jl root:
#   bash benchmark/probprog/ablation/pre_lowering/run_prelowering_batch_v4.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DIR="$ROOT/benchmark/probprog/ablation/pre_lowering"
LOGS="$DIR/logs"
SO="$ROOT/deps/ReactantExtra/bazel-bin/libReactantExtra.so"
META="$LOGS/batch_meta.txt"
LOAD_GATE="${PRELOWER_LOAD_GATE:-8}"
WARMUP="${PRELOWER_WARMUP:-200}"
export PRELOWER_WARMUP="$WARMUP"

mkdir -p "$LOGS" "$LOGS/cells" "$LOGS/hlo" "$DIR/adapted_states"

SO_MD5="$(md5sum "$SO" | awk '{print $1}')"
{
  echo "batch start: $(date -Is)"
  echo "protocol: frozen-adapted-state (warmup adaptation ON in arm B, both arms measured adaptation OFF from that state)"
  echo "warmup iterations: $WARMUP"
  echo "so mtime: $(stat -c '%y %s' "$SO")"
  echo "so md5:   $SO_MD5"
  echo "loadavg:  $(cat /proc/loadavg)"
  echo "nproc:    $(nproc)"
  echo "julia:    $(julia --version)"
  echo "load gate: load1 < $LOAD_GATE and no pplbench/harness.jl, NO ceiling"
} > "$META"

sibling_count () {
  pgrep -af 'pplbench|harness\.jl' 2>/dev/null | grep -v 'zsh -c' | grep -cv pgrep || true
}

wait_for_quiet () {
  local waited=0 busy load1
  while :; do
    busy="$(sibling_count)"
    load1="$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)"
    if [ "$busy" -eq 0 ] && [ "$load1" -lt "$LOAD_GATE" ]; then
      [ "$waited" -gt 0 ] && \
        echo "    (waited ${waited}s for quiet; load1=$load1 sibling=$busy)" >> "$META"
      return 0
    fi
    if [ $((waited % 600)) -eq 0 ]; then
      echo "    waiting for quiet: ${waited}s elapsed, sibling=$busy load1=$load1 $(date -Is)" >> "$META"
    fi
    sleep 30
    waited=$((waited + 30))
  done
}

run_cell () {   # $1 = tag, rest = julia script + args
  local tag="$1"; shift
  wait_for_quiet
  echo "--- cell $tag  load_before=$(cut -d' ' -f1-3 /proc/loadavg)  sibling=$(sibling_count)  $(date -Is)" >> "$META"
  ( cd "$ROOT" && timeout 7200 script -q -c \
      "JULIA_CI=true CUDA_VISIBLE_DEVICES='' PRELOWER_WARMUP=$WARMUP julia --project=benchmark/probprog $*" \
      "$LOGS/${tag}.log" > /dev/null 2>&1 )
  local rc=$?
  local now_md5 sib_after
  now_md5="$(md5sum "$SO" | awk '{print $1}')"
  [ "$now_md5" != "$SO_MD5" ] && \
    echo "!! BINARY CHANGED during $tag: $SO_MD5 -> $now_md5" >> "$META"
  sib_after="$(sibling_count)"
  [ "$sib_after" -ne 0 ] && \
    echo "!! SIBLING APPEARED DURING $tag (count=$sib_after): cell is suspect" >> "$META"
  echo "    rc=$rc  md5=$now_md5  sibling_after=$sib_after  load_after=$(cut -d' ' -f1-3 /proc/loadavg)  $(date -Is)" >> "$META"
  grep -E "Median|Geomean|All:|verification|stability|adapted step|frozen-adapted|step=" \
    "$LOGS/${tag}.log" >> "$META"
}

S="benchmark/probprog/ablation/pre_lowering"

# Per model: warmup (arm B, adaptation ON) then the two measured arms adjacent
# in time so each ratio's arms see the same machine conditions.
run_cell heat_s1_warmup  "$S/heat_inverse.jl B 1 warmup"
run_cell heat_s1_B       "$S/heat_inverse.jl B 1 bench"
run_cell heat_s1_A       "$S/heat_inverse.jl A 1 bench"

run_cell heat_s3_warmup  "$S/heat_inverse.jl B 3 warmup"
run_cell heat_s3_B       "$S/heat_inverse.jl B 3 bench"
run_cell heat_s3_A       "$S/heat_inverse.jl A 3 bench"

run_cell advdiff_warmup  "$S/advection_diffusion.jl B 1 warmup"
run_cell advdiff_B       "$S/advection_diffusion.jl B 1 bench"
run_cell advdiff_A       "$S/advection_diffusion.jl A 1 bench"

run_cell star_warmup     "$S/star_ili.jl B warmup"
run_cell star_B          "$S/star_ili.jl B bench"
run_cell star_A          "$S/star_ili.jl A bench"

{
  echo "batch end: $(date -Is)"
  echo "so md5:   $(md5sum "$SO" | awk '{print $1}')"
  echo "loadavg:  $(cat /proc/loadavg)"
} >> "$META"
echo "done"
