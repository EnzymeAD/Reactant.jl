#!/usr/bin/env bash
# Controlled single-build batch for tab:prelowering (2026-08-24).
#
# One uniform protocol for every row: flat expand path in both arms, SICM off in
# both arms, arms differ ONLY in disable_licm_optimization_passes.
# 20 NUTS samples, adaptation off, CPU, medians of 5 runs.
#
# Run from the Reactant.jl root:
#   bash benchmark/probprog/ablation/pre_lowering/run_prelowering_batch.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DIR="$ROOT/benchmark/probprog/ablation/pre_lowering"
LOGS="$DIR/logs"
SO="$ROOT/deps/ReactantExtra/bazel-bin/libReactantExtra.so"
META="$LOGS/batch_meta.txt"

mkdir -p "$LOGS" "$LOGS/cells" "$LOGS/hlo" "$LOGS/parity"

{
  echo "batch start: $(date -Is)"
  echo "so mtime: $(stat -c '%y %s' "$SO")"
  echo "so md5:   $(md5sum "$SO" | awk '{print $1}')"
  echo "loadavg:  $(cat /proc/loadavg)"
  echo "nproc:    $(nproc)"
  echo "julia:    $(julia --version)"
} > "$META"

run_cell () {   # $1 = tag, rest = julia script + args
  local tag="$1"; shift
  echo "--- cell $tag  load_before=$(cut -d' ' -f1-3 /proc/loadavg)  $(date -Is)" >> "$META"
  ( cd "$ROOT" && timeout 7200 script -q -c \
      "JULIA_CI=true CUDA_VISIBLE_DEVICES='' julia --project=benchmark/probprog $*" \
      "$LOGS/${tag}.log" > /dev/null 2>&1 )
  local rc=$?
  echo "    rc=$rc  load_after=$(cut -d' ' -f1-3 /proc/loadavg)  $(date -Is)" >> "$META"
  grep -E "Median|All:|verification|n_accepted" "$LOGS/${tag}.log" >> "$META"
}

S="benchmark/probprog/ablation/pre_lowering"

# ── Table cells: B (Pre-Opt) then A (No Pre-Opt), adjacent in time per model ──
run_cell cubic_small_B     "$S/cubic_kernel_dynamics.jl B small bench"
run_cell cubic_small_A     "$S/cubic_kernel_dynamics.jl A small bench"
run_cell cubic_large_B     "$S/cubic_kernel_dynamics.jl B large bench"
run_cell cubic_large_A     "$S/cubic_kernel_dynamics.jl A large bench"
run_cell projected_B       "$S/projected_dynamics.jl B bench"
run_cell projected_A       "$S/projected_dynamics.jl A bench"
run_cell heat_s1_B         "$S/heat_inverse.jl B 1 bench"
run_cell heat_s1_A         "$S/heat_inverse.jl A 1 bench"
run_cell heat_s3_B         "$S/heat_inverse.jl B 3 bench"
run_cell heat_s3_A         "$S/heat_inverse.jl A 3 bench"

# ── Heat cross-arm verification (samples + accept decisions, seed [1,5]) ──────
# heat_inverse.jl parity mode writes one file per scale; copy it out per arm.
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
