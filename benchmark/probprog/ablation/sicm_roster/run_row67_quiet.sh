#!/bin/bash
# One quiet-window pass for the roster extension:
#   1. smoke the Columbus SAR fallback (n=49)
#   2. dump-first IR for both Columbus arms
#   3. measured batch: Columbus SAR + a quiet re-run of PGLS
#
# PGLS is re-run because its first measurement (batch row67a) was taken under
# load1 9.7 -> 43.9 from an external user's julia; both figure statistics should
# come from a quiet window. Both cells are subsecond-to-second per arm, so the
# whole pass costs a few minutes of machine.
set -uo pipefail

BENCH_DIR="/mnt/sbrantq/probprog/oopsla/Reactant.jl/benchmark/probprog"
SO="/mnt/sbrantq/probprog/oopsla/Reactant.jl/deps/ReactantExtra/bazel-bin/libReactantExtra.so"
LOG_DIR="$BENCH_DIR/ablation/sicm_roster/logs_row67quiet"
HLO_DIR="$BENCH_DIR/ablation/sicm_roster/hlo"
mkdir -p "$LOG_DIR" "$HLO_DIR"
cd "$BENCH_DIR" || exit 1

echo "md5 at start: $(md5sum "$SO" | cut -d' ' -f1)"
echo "loadavg at start: $(cat /proc/loadavg)"

echo "=== 1. smoke: Columbus SAR ==="
CUDA_VISIBLE_DEVICES="" timeout 1800 \
  script -q -c "./run_pplbench.sh --threads all probe/sicm_smoke_sar_columbus" \
  "$LOG_DIR/smoke_sar_columbus.log" >/dev/null 2>&1
tr -d '\r' < "$LOG_DIR/smoke_sar_columbus.log" \
  | grep -oE "inference trial [01] took [0-9.]* secs|Output dir: [^ ]*|COMMAND_EXIT_CODE=\"[0-9]*\""

echo "=== 2. dump-first IR: Columbus SAR, both arms ==="
for arm in sicm nosicm; do
  cfg="probe/sicm_ir_sar_columbus_${arm}"
  log="$LOG_DIR/ir_sar_columbus_${arm}.log"
  CUDA_VISIBLE_DEVICES="" PPLBENCH_DUMP_MLIR_ALWAYS=1 timeout 3600 \
    script -q -c "./run_pplbench.sh --threads all --dump-mlir $cfg" "$log" >/dev/null 2>&1
  outdir=$(grep -m1 'Output dir:' "$log" | sed 's/.*Output dir: //' | tr -d '\r')
  src=$(ls "$outdir"/mlir_dumps/*pre_xla_compile.mlir 2>/dev/null | head -1)
  src2=$(ls "$outdir"/mlir_dumps/*post_impulse_pm.mlir 2>/dev/null | head -1)
  [ -n "$src2" ] && sed -n '1,2p' "$src2" > "$HLO_DIR/sar_columbus_${arm}_pipeline.txt"
  if [ -n "$src" ]; then
    cp "$src" "$HLO_DIR/sar_columbus_${arm}.mlir"
    echo "  kept $(du -h "$HLO_DIR/sar_columbus_${arm}.mlir" | cut -f1) -> hlo/sar_columbus_${arm}.mlir"
  else
    echo "  !! no pre_xla_compile dump for $arm"
  fi
  rm -rf "$outdir/mlir_dumps"
done

echo "=== 3. measured batch: Columbus SAR + PGLS re-run ==="
CELLS="sicm_ablation/sar_columbus sicm_ablation/pgls" \
  bash "$BENCH_DIR/ablation/sicm_roster/run_batch.sh" row67quiet

echo "md5 at end: $(md5sum "$SO" | cut -d' ' -f1)"
echo "loadavg at end: $(cat /proc/loadavg)"
echo done
