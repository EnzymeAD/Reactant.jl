#!/bin/bash
# Uncapped (ablation-regime) batches of the 2026-09-08 re-freeze, run strictly
# one at a time. Each driver has its own quiet gate (ambient load1 < 8, external
# CPU < 300% to start; external CPU sampled every 5 s during every cell with
# automatic retry), so this script only sequences them.
#
#   1. pre-lowering (fig "prelowering"): heat x2, advection-diffusion, STAR
#      + the cubic-kernel and projected-dynamics rows, frozen adapted state
#   2. SICM roster (fig "sicm-ablation"): the six plotted rows + the section-2
#      phylogenetic Poisson regression row
#   3. SICM-on-VI: three repetitions of the n=200 driver (median over runs)
#   4. IR evidence for the section-2 row (in-loop LAPACK call counts)
set -uo pipefail
BENCH="/mnt/sbrantq/probprog/oopsla/Reactant.jl/benchmark/probprog"
ROOT="/mnt/sbrantq/probprog/oopsla/Reactant.jl"
EV="/mnt/sbrantq/probprog/oopsla/evidence/refreeze_2026-09-08"
LOGS="$BENCH/logs/refreeze_2026-09-08"
cd "$BENCH" || exit 1
mkdir -p "$LOGS/uncapped" "$EV"
DRV="$LOGS/uncapped/driver.log"
note() { echo "[$(date '+%F %T')] $*" | tee -a "$DRV"; }

STAGES="${STAGES:-prelower_main cubic_warmup cubic_bench roster svi ir}"
note "uncapped phase start; stages: $STAGES"

for st in $STAGES; do
case "$st" in

prelower_main)
  note "STAGE prelower_main: ${PRELOWER_MAIN_CELLS:-heat_s1 heat_s3 advdiff star}"
  PRELOWER_LOAD_GATE="${PRELOWER_LOAD_GATE:-8}" \
  PRELOWER_START_EXT_CPU="${PRELOWER_START_EXT_CPU:-700}" \
  PRELOWER_EXT_CPU_LIMIT="${PRELOWER_EXT_CPU_LIMIT:-1000}" \
  PRELOWER_MAX_RETRIES="${PRELOWER_MAX_RETRIES:-60}" \
  PRELOWER_CELLS="${PRELOWER_MAIN_CELLS:-heat_s1 heat_s3 advdiff star}" \
    bash "$BENCH/ablation/pre_lowering/run_prelowering_batch_v5.sh" \
    > "$LOGS/uncapped/prelower_main.log" 2>&1
  note "STAGE prelower_main rc=$?"
  ;;

cubic_warmup)
  note "STAGE cubic_warmup: mint adapted states for cubic small/large and projected"
  PRELOWER_LOAD_GATE="${PRELOWER_LOAD_GATE:-8}" \
  PRELOWER_START_EXT_CPU="${PRELOWER_START_EXT_CPU:-700}" \
  PRELOWER_EXT_CPU_LIMIT="${PRELOWER_EXT_CPU_LIMIT:-1000}" \
  PRELOWER_MAX_RETRIES="${PRELOWER_MAX_RETRIES:-60}" \
  PRELOWER_CELLS="cubic_s1_w cubic_s2_w projected_w" \
    bash "$BENCH/ablation/pre_lowering/run_prelowering_batch_v5.sh" \
    > "$LOGS/uncapped/cubic_warmup.log" 2>&1
  note "STAGE cubic_warmup rc=$?"
  ;;

cubic_bench)
  note "STAGE cubic_bench: cubic small/large + projected, both arms"
  PRELOWER_LOAD_GATE="${PRELOWER_LOAD_GATE:-8}" \
  PRELOWER_START_EXT_CPU="${PRELOWER_START_EXT_CPU:-700}" \
  PRELOWER_EXT_CPU_LIMIT="${PRELOWER_EXT_CPU_LIMIT:-1000}" \
  PRELOWER_MAX_RETRIES="${PRELOWER_MAX_RETRIES:-60}" \
  PRELOWER_CELLS="cubic_s1 cubic_s2 projected" \
    bash "$BENCH/ablation/pre_lowering/run_prelowering_batch_v5.sh" \
    > "$LOGS/uncapped/cubic_bench.log" 2>&1
  note "STAGE cubic_bench rc=$?"
  ;;

roster)
  note "STAGE roster: six plotted rows + phylo_pois"
  FROZEN_LOAD_GATE="${FROZEN_LOAD_GATE:-8}" \
  FROZEN_START_EXT_CPU="${FROZEN_START_EXT_CPU:-700}" \
  FROZEN_EXT_CPU_LIMIT="${FROZEN_EXT_CPU_LIMIT:-1000}" \
  FROZEN_MAX_RETRIES="${FROZEN_MAX_RETRIES:-60}" \
  FROZEN_CELLS="
car_spatial_scotland:sicm/car_spatial:0.05:10
car_spatial_nyc:sicm/car_spatial:0.05:10
pgls:sicm/pgls:0.05:10
hierarchical_mvn:sicm/hierarchical_mvn:0.01:10
lmm_radon:sicm/linear_mixed_model:0.05:10
diamonds:sicm/diamonds_regression:0.05:10
phylo_pois:motivating/phylo_pois:0.1:10
" bash "$BENCH/ablation/sicm_roster/run_frozen_batch.sh" refreeze_2026-09-08 \
    > "$LOGS/uncapped/roster.log" 2>&1
  note "STAGE roster rc=$?"
  ;;

svi)
  note "STAGE svi: three repetitions of the n=200 SICM-on-VI driver"
  # Through refreeze_cell.sh so each repetition gets the same quiet-window gate,
  # per-cell load sampling, core-steal record and retry loop as every other cell.
  #
  # ABORT_SAMPLES defaults to 24 here, not 1. refreeze_cell.sh accepts a cell on
  # its SUSTAINED (five-minute mean) external CPU, so aborting on a single
  # instantaneous sample over the limit throws away cells that would have
  # passed: svi_run1 was killed at 100 s on one 1117% sample while its sustained
  # figure was 419%. The uncapped batch drivers are the opposite case, since
  # they accept on the instantaneous maximum, and there aborting on the first
  # violating sample is exactly right.
  for r in 1 2 3; do
    mkdir -p "$EV/sicm_on_vi/run$r" "$LOGS/svi/cells"
    LOAD_GATE="${FROZEN_LOAD_GATE:-12}" START_EXT_CPU="${FROZEN_START_EXT_CPU:-900}" \
    LOAD_MAX="${FROZEN_LOAD_GATE:-12}" EXT_CPU_LIMIT="${FROZEN_EXT_CPU_LIMIT:-1000}" \
    ABORT_SAMPLES="${SVI_ABORT_SAMPLES:-24}" MAX_RETRIES="${FROZEN_MAX_RETRIES:-400}" \
      "$BENCH/refreeze_cell.sh" "svi_run$r" "$LOGS/svi" 2h none -- \
      "cd $ROOT && CUDA_VISIBLE_DEVICES= D3_OUT=$EV/sicm_on_vi/run$r julia --project=test test/probprog/svi_sicm_benchmark.jl" \
      >> "$LOGS/uncapped/svi_run$r.driver.log" 2>&1
    note "  svi run$r rc=$?"
  done
  ;;

ir)
  note "STAGE ir: section-2 phylogenetic Poisson regression IR dumps"
  IR_MODELS="phylo_pois" bash - <<'IRSH' > "$LOGS/uncapped/ir_phylo.log" 2>&1
BENCH="/mnt/sbrantq/probprog/oopsla/Reactant.jl/benchmark/probprog"
HLO="$BENCH/ablation/sicm_roster/hlo"
LOG="$BENCH/logs/refreeze_2026-09-08/uncapped"
mkdir -p "$HLO" "$LOG"
cd "$BENCH" || exit 1
for arm in sicm nosicm; do
  cfg="motivating/phylo_pois_ir_${arm}"
  log="$LOG/ir_phylo_${arm}.log"
  echo "=== $cfg ==="
  CUDA_VISIBLE_DEVICES="" PPLBENCH_DUMP_MLIR_ALWAYS=1 timeout 7200 \
    script -q -c "./run_pplbench.sh --threads all --dump-mlir $cfg" "$log"
  outdir=$(grep -a -m1 'Output dir:' "$log" | sed 's/.*Output dir: //' | tr -d '\r')
  echo "  outdir=$outdir"
  src=$(ls "$outdir"/mlir_dumps/*pre_xla_compile.mlir 2>/dev/null | head -1)
  src2=$(ls "$outdir"/mlir_dumps/*post_impulse_pm.mlir 2>/dev/null | head -1)
  [ -n "$src2" ] && sed -n '1,2p' "$src2" > "$HLO/phylo_pois_${arm}_pipeline.txt"
  if [ -n "$src" ]; then
    cp "$src" "$HLO/phylo_pois_${arm}.mlir"
    echo "  kept $(du -h "$HLO/phylo_pois_${arm}.mlir" | cut -f1)"
  else
    echo "  !! no pre_xla_compile dump in $outdir"
  fi
  rm -rf "$outdir/mlir_dumps"
done
IRSH
  note "STAGE ir rc=$?"
  ;;

*) note "!! unknown stage $st" ;;
esac
done
note "uncapped phase done"
