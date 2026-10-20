#!/bin/bash
# IR evidence: one single-arm tiny run per (model, arm), each into its own
# output dir so the module dumps of the two arms never mix.
#
# Keeps, per cell:
#   hlo/<model>_<arm>.mlir              the final IR (`pre_xla_compile`)
#   hlo/<model>_<arm>_pipeline.txt      the pass pipeline string recorded in the
#                                       `post_impulse_pm` dump, which shows
#                                       `sicm{...}` vs `sicm{... max-iterations=0}`
# and deletes the rest of the dump directory (tens of MB per cell).
set -uo pipefail

BENCH_DIR="/mnt4/sbrantq/probprog/oopsla/Reactant.jl/benchmark/probprog"
LOG_DIR="$BENCH_DIR/ablation/sicm_roster/logs_ir"
HLO_DIR="$BENCH_DIR/ablation/sicm_roster/hlo"
mkdir -p "$LOG_DIR" "$HLO_DIR"

cd "$BENCH_DIR" || exit 1

echo "md5 at start: $(md5sum /mnt4/sbrantq/probprog/oopsla/Reactant.jl/deps/ReactantExtra/bazel-bin/libReactantExtra.so | cut -d' ' -f1)"

for m in ${IR_MODELS:-car_spatial_scotland car_spatial_nyc lmm_radon hierarchical_mvn}; do
  for arm in sicm nosicm; do
    cfg="probe/sicm_ir_${m}_${arm}"
    log="$LOG_DIR/${m}_${arm}.log"
    echo "=== $cfg ==="
    CUDA_VISIBLE_DEVICES="" PPLBENCH_DUMP_MLIR_ALWAYS=1 timeout 7200 \
        script -q -c "./run_pplbench.sh --threads all --dump-mlir $cfg" "$log"
    outdir=$(grep -m1 'Output dir:' "$log" | sed 's/.*Output dir: //' | tr -d '\r')
    echo "  outdir=$outdir"
    src=$(ls "$outdir"/mlir_dumps/*pre_xla_compile.mlir 2>/dev/null | head -1)
    src2=$(ls "$outdir"/mlir_dumps/*post_impulse_pm.mlir 2>/dev/null | head -1)
    if [ -n "$src2" ]; then
      sed -n '1,2p' "$src2" > "$HLO_DIR/${m}_${arm}_pipeline.txt"
    fi
    if [ -n "$src" ]; then
      cp "$src" "$HLO_DIR/${m}_${arm}.mlir"
      echo "  kept $(du -h "$HLO_DIR/${m}_${arm}.mlir" | cut -f1) -> hlo/${m}_${arm}.mlir"
    else
      echo "  !! no pre_xla_compile dump found in $outdir"
    fi
    rm -rf "$outdir/mlir_dumps"
  done
done
echo "md5 at end: $(md5sum /mnt4/sbrantq/probprog/oopsla/Reactant.jl/deps/ReactantExtra/bazel-bin/libReactantExtra.so | cut -d' ' -f1)"
echo done
