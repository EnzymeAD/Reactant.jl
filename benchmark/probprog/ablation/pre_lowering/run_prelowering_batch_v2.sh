#!/usr/bin/env bash
# Controlled single-build batch for the FOUR figure rows of tab:prelowering
# (2026-08-25): Heat, Heat (large), Advection-Diffusion, STAR.
#
# Same two-arm protocol as run_prelowering_batch.sh (which measured the now
# superseded Cubic Kernel / Projected Dynamics rows): flat expand path in both
# arms, SICM off in both arms, arms differ ONLY in
# disable_licm_optimization_passes. 20 NUTS samples, adaptation off, CPU,
# median of 5 runs per cell after one warm-up call, no thread caps.
#
# Run from the Reactant.jl root:
#   bash benchmark/probprog/ablation/pre_lowering/run_prelowering_batch_v2.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DIR="$ROOT/benchmark/probprog/ablation/pre_lowering"
LOGS="$DIR/logs"
SO="$ROOT/deps/ReactantExtra/bazel-bin/libReactantExtra.so"
META="$LOGS/batch_meta.txt"

mkdir -p "$LOGS" "$LOGS/cells" "$LOGS/hlo" "$LOGS/parity"

SO_MD5="$(md5sum "$SO" | awk '{print $1}')"
{
  echo "batch start: $(date -Is)"
  echo "so mtime: $(stat -c '%y %s' "$SO")"
  echo "so md5:   $SO_MD5"
  echo "loadavg:  $(cat /proc/loadavg)"
  echo "nproc:    $(nproc)"
  echo "julia:    $(julia --version)"
} > "$META"

# Serialize behind the sibling SICM campaign: wait until no pplbench / harness
# measurement is in flight and load1 has come down.
wait_for_quiet () {
  local tries=0
  while :; do
    local busy load1
    busy="$(pgrep -af 'pplbench|harness\.jl' | grep -v 'zsh -c' | grep -cv pgrep || true)"
    load1="$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)"
    if [ "$busy" -eq 0 ] && [ "$load1" -lt 12 ]; then
      return 0
    fi
    tries=$((tries + 1))
    if [ "$tries" -gt 120 ]; then      # 60 min ceiling, then report and proceed
      echo "!! contention persisted 60 min (sibling=$busy load1=$load1); proceeding" >> "$META"
      return 1
    fi
    sleep 30
  done
}

run_cell () {   # $1 = tag, rest = julia script + args
  local tag="$1"; shift
  wait_for_quiet
  local sib
  sib="$(pgrep -af 'pplbench|harness\.jl' | grep -v 'zsh -c' | grep -cv pgrep || true)"
  echo "--- cell $tag  load_before=$(cut -d' ' -f1-3 /proc/loadavg)  sibling=$sib  $(date -Is)" >> "$META"
  ( cd "$ROOT" && timeout 7200 script -q -c \
      "JULIA_CI=true CUDA_VISIBLE_DEVICES='' julia --project=benchmark/probprog $*" \
      "$LOGS/${tag}.log" > /dev/null 2>&1 )
  local rc=$?
  # Per-cell binary integrity: the whole table must be one build.
  local now_md5
  now_md5="$(md5sum "$SO" | awk '{print $1}')"
  if [ "$now_md5" != "$SO_MD5" ]; then
    echo "!! BINARY CHANGED during $tag: $SO_MD5 -> $now_md5" >> "$META"
  fi
  echo "    rc=$rc  md5=$now_md5  load_after=$(cut -d' ' -f1-3 /proc/loadavg)  $(date -Is)" >> "$META"
  grep -E "Median|All:|verification|n_accepted|stability" "$LOGS/${tag}.log" >> "$META"
}

S="benchmark/probprog/ablation/pre_lowering"

# ── Figure cells: B (Pre-Opt) then A (No Pre-Opt), adjacent in time per model ─
run_cell heat_s1_B    "$S/heat_inverse.jl B 1 bench"
run_cell heat_s1_A    "$S/heat_inverse.jl A 1 bench"
run_cell heat_s3_B    "$S/heat_inverse.jl B 3 bench"
run_cell heat_s3_A    "$S/heat_inverse.jl A 3 bench"
run_cell advdiff_B    "$S/advection_diffusion.jl B 1 bench"
run_cell advdiff_A    "$S/advection_diffusion.jl A 1 bench"
run_cell star_B       "$S/star_ili.jl B bench"
run_cell star_A       "$S/star_ili.jl A bench"

# ── Heat cross-arm verification (samples + accept decisions, seed [1,5]) ──────
# advdiff and star fold their verification run into `bench`; heat_inverse.jl is
# unmodified, so its verification uses the existing `parity` mode.
for sc in 1 3; do
  for arm in B A; do
    run_cell "heat_parity_s${sc}_${arm}" "$S/heat_inverse.jl $arm $sc parity"
    cp "/tmp/sbrantq-heatinv/phase1/heat_samples_julia_scale${sc}.json" \
       "$LOGS/parity/heat_s${sc}_arm${arm}.json" 2>/dev/null
  done
done

{
  echo "batch end: $(date -Is)"
  echo "so md5:   $(md5sum "$SO" | awk '{print $1}')"
  echo "loadavg:  $(cat /proc/loadavg)"
} >> "$META"
echo "done"
