#!/bin/bash
# One contemporaneous single-build batch for the four sicm_ablation figure rows.
#
#   1. CholScale      car_spatial_scotland (real Scotland lip cancer, n=56)
#   2. CholScale      car_spatial_nyc      (posteriordb NYC traffic sub-map, n=250)
#   3. OuterProdScale hierarchical_mvn     (K=60, J=2; config run exactly as it stands)
#   4. EigenLift      lmm_radon            (real Minnesota radon, n=250 subsample)
#
# Override the cell list with CELLS="sicm_ablation/x sicm_ablation/y".
#
# Fully uncapped (ablation regime): no taskset, run_pplbench.sh --threads all.
# Records the libReactantExtra.so md5 and /proc/loadavg before and after every
# cell, per the single-build rule.
set -uo pipefail

BENCH_DIR="/mnt/sbrantq/probprog/oopsla/Reactant.jl/benchmark/probprog"
SO="/mnt/sbrantq/probprog/oopsla/Reactant.jl/deps/ReactantExtra/bazel-bin/libReactantExtra.so"
TAG="${1:-batch1}"
LOG_DIR="$BENCH_DIR/ablation/sicm_roster/logs_$TAG"
META="$LOG_DIR/batch_meta.txt"
mkdir -p "$LOG_DIR"

cd "$BENCH_DIR" || exit 1

{
  echo "batch tag: $TAG"
  echo "start: $(date -Is)"
  echo "binary: $SO"
  echo "md5 at batch start: $(md5sum "$SO" | cut -d' ' -f1)"
  echo "mtime: $(stat -c %y "$SO")"
  echo "loadavg at batch start: $(cat /proc/loadavg)"
  echo "sibling taskset processes: $(pgrep -af taskset | tr '\n' '|')"
  echo "nproc: $(getconf _NPROCESSORS_ONLN)"
} > "$META"

for cfg in ${CELLS:-sicm_ablation/car_spatial_scotland \
           sicm_ablation/car_spatial_nyc \
           sicm_ablation/lmm_radon \
           sicm_ablation/hierarchical_mvn}; do
    name="${cfg##*/}"
    {
      echo "--- cell $name"
      echo "  start: $(date -Is)"
      echo "  loadavg before: $(cat /proc/loadavg)"
      echo "  md5 before: $(md5sum "$SO" | cut -d' ' -f1)"
    } >> "$META"

    CUDA_VISIBLE_DEVICES="" timeout 21600 \
        script -q -c "./run_pplbench.sh --threads all $cfg" \
        "$LOG_DIR/${name}.log"

    {
      echo "  end: $(date -Is)"
      echo "  loadavg after: $(cat /proc/loadavg)"
      echo "  md5 after: $(md5sum "$SO" | cut -d' ' -f1)"
      echo "  output dir: $(grep -m1 'Output dir:' "$LOG_DIR/${name}.log" | sed 's/.*Output dir: //')"
      echo "  script exit: $(grep -o 'COMMAND_EXIT_CODE=\"[0-9]*\"' "$LOG_DIR/${name}.log" | tail -1)"
    } >> "$META"
done

{
  echo "end: $(date -Is)"
  echo "md5 at batch end: $(md5sum "$SO" | cut -d' ' -f1)"
  echo "loadavg at batch end: $(cat /proc/loadavg)"
} >> "$META"

cat "$META"
