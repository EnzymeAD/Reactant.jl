#!/usr/bin/env bash
# Controlled single-build batch for the two NEW figure rows of tab:prelowering
# (2026-08-25): Advection-Diffusion and STAR.
#
# The Heat rows are NOT re-measured here. Per the 2026-08-25 arbitration they
# stand from the quiet 2026-08-24 batch 2 (logs_batch2/), which ran on the same
# libReactantExtra.so md5 under load1 2.5-3.4. The figure is therefore
# single-BUILD and two-batch, with load conditions recorded for both.
#
# Protocol identical to run_prelowering_batch.sh: flat expand path in both arms,
# SICM off in both arms, arms differ ONLY in disable_licm_optimization_passes.
# 20 NUTS samples, adaptation off, CPU, median of 5 runs after one warm-up.
#
# Machine discipline (2026-08-25 directive): NEVER measure under load. The gate
# below waits indefinitely for the sibling SICM campaign to clear; there is no
# proceed-anyway ceiling. A late clean number beats an on-time dirty one.
#
# Run from the Reactant.jl root:
#   bash benchmark/probprog/ablation/pre_lowering/run_prelowering_batch_v3.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DIR="$ROOT/benchmark/probprog/ablation/pre_lowering"
LOGS="$DIR/logs"
SO="$ROOT/deps/ReactantExtra/bazel-bin/libReactantExtra.so"
META="$LOGS/batch_meta.txt"
LOAD_GATE="${PRELOWER_LOAD_GATE:-8}"

mkdir -p "$LOGS" "$LOGS/cells" "$LOGS/hlo"

SO_MD5="$(md5sum "$SO" | awk '{print $1}')"
{
  echo "batch start: $(date -Is)"
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

# Wait indefinitely for a genuinely quiet machine. No proceed-anyway ceiling.
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
      echo "    waiting for quiet: ${waited}s elapsed, sibling=$busy load1=$load1 $(date -Is)" \
        >> "$META"
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
      "JULIA_CI=true CUDA_VISIBLE_DEVICES='' julia --project=benchmark/probprog $*" \
      "$LOGS/${tag}.log" > /dev/null 2>&1 )
  local rc=$?
  # Per-cell binary integrity: this table and the SICM table must share a build.
  local now_md5
  now_md5="$(md5sum "$SO" | awk '{print $1}')"
  if [ "$now_md5" != "$SO_MD5" ]; then
    echo "!! BINARY CHANGED during $tag: $SO_MD5 -> $now_md5" >> "$META"
  fi
  # Did a sibling job appear WHILE the cell ran? If so the cell is suspect.
  local sib_after
  sib_after="$(sibling_count)"
  [ "$sib_after" -ne 0 ] && \
    echo "!! SIBLING APPEARED DURING $tag (count=$sib_after): cell is suspect" >> "$META"
  echo "    rc=$rc  md5=$now_md5  sibling_after=$sib_after  load_after=$(cut -d' ' -f1-3 /proc/loadavg)  $(date -Is)" >> "$META"
  grep -E "Median|All:|verification|stability" "$LOGS/${tag}.log" >> "$META"
}

S="benchmark/probprog/ablation/pre_lowering"

# B (Pre-Opt) then A (No Pre-Opt), adjacent in time so each ratio's two arms
# see the same machine conditions.
run_cell advdiff_B "$S/advection_diffusion.jl B 1 bench"
run_cell advdiff_A "$S/advection_diffusion.jl A 1 bench"
run_cell star_B    "$S/star_ili.jl B bench"
run_cell star_A    "$S/star_ili.jl A bench"

{
  echo "batch end: $(date -Is)"
  echo "so md5:   $(md5sum "$SO" | awk '{print $1}')"
  echo "loadavg:  $(cat /proc/loadavg)"
} >> "$META"
echo "done"
