#!/bin/bash
# Smoke the three new/refreshed sicm_ablation configs at tiny iteration counts.
# Functional only: does the pipeline run end to end, do both arms produce
# samples, do the chains move.
set -uo pipefail

BENCH_DIR="/mnt/sbrantq/probprog/oopsla/Reactant.jl/benchmark/probprog"
LOG_DIR="$BENCH_DIR/ablation/sicm_roster/logs_smoke"
mkdir -p "$LOG_DIR"

cd "$BENCH_DIR" || exit 1

echo "binary md5: $(md5sum /mnt/sbrantq/probprog/oopsla/Reactant.jl/deps/ReactantExtra/bazel-bin/libReactantExtra.so)"
echo "loadavg before all: $(cat /proc/loadavg)"

for cfg in ${SMOKE_CELLS:-probe/sicm_smoke_car_scotland probe/sicm_smoke_car_nyc \
           probe/sicm_smoke_lmm_radon probe/sicm_smoke_phylo_pois}; do
    name="${cfg##*/}"
    echo "=== $cfg ==="
    echo "loadavg before: $(cat /proc/loadavg)"
    CUDA_VISIBLE_DEVICES="" timeout 3600 \
        script -q -c "./run_pplbench.sh --threads all $cfg" \
        "$LOG_DIR/smoke_${name}.log"
    echo "exit=$? loadavg after: $(cat /proc/loadavg)"
done

echo "binary md5 after: $(md5sum /mnt/sbrantq/probprog/oopsla/Reactant.jl/deps/ReactantExtra/bazel-bin/libReactantExtra.so)"
echo "loadavg after all: $(cat /proc/loadavg)"
