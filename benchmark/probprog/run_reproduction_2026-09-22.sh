#!/bin/bash
# Reproduction of every Impulse-side cell of the 2026-09-08 re-freeze and the
# 2026-09-10 size sweep on the rebased binary of 2026-09-22 (with the syevd
# heap-workspace fix). Same drivers, same protocols, same gates; this script
# only sequences them under a new campaign tag.
#
#   phase "uncapped": each cell alone (drivers own the gates: start at
#     procs_running < 8, external CPU < 700% for 30 s, no bazel build active;
#     strict instantaneous acceptance at external CPU < 1000%).
#       fig 9 pre-lowering rows, fig 10 SICM roster rows at the plotted sizes
#       (+ Radon n=500 and n=919) and the section-2 row, SICM-on-VI x3, section-2 IR.
#   phase "pinned": one core per CCD, concurrently:
#       fig 11 Impulse cells (core 8), fig 12 Impulse variants (cores 32 40 48),
#       fig 13 Impulse cells + isolated-SICM triples (core 56), coal (core 64).
#
# Usage: run_reproduction_2026-09-22.sh [uncapped] [pinned]
set -uo pipefail
BENCH="/mnt/sbrantq/probprog/oopsla/Reactant.jl/benchmark/probprog"
EV="/mnt/sbrantq/probprog/oopsla/evidence/rebase_2026-09-22"
CAMPAIGN="rebase_2026-09-22"
LOGS="$BENCH/logs/$CAMPAIGN"
SO="/mnt/sbrantq/probprog/oopsla/Reactant.jl/deps/ReactantExtra/bazel-bin/libReactantExtra.so"
EXPECT_MD5="51db760bd3f057d17a7413cd5400886d"
cd "$BENCH" || exit 1
mkdir -p "$LOGS" "$EV"
DRV="$LOGS/supervisor.log"
note() { echo "[$(date '+%F %T')] $*" | tee -a "$DRV"; }
check_md5() {
  local m; m="$(md5sum "$SO" | cut -d' ' -f1)"
  note "md5 $m"
  [ "$m" = "$EXPECT_MD5" ] || { note "!! binary md5 $m != $EXPECT_MD5; stopping"; exit 3; }
}
export STRICT=1
PHASES="${*:-uncapped pinned}"
note "reproduction start; phases: $PHASES"
check_md5

for ph in $PHASES; do
case "$ph" in
uncapped)
  if [ "${SKIP_PRELOWER:-0}" != "1" ]; then
    note "STAGE fig09 pre-lowering: heat_s1 heat_s3 advdiff star"
    PRELOWER_LOAD_GATE=8 PRELOWER_START_EXT_CPU=700 PRELOWER_EXT_CPU_LIMIT=1000 \
    PRELOWER_MAX_RETRIES=60 PRELOWER_CELLS="heat_s1 heat_s3 advdiff star" \
      bash "$BENCH/ablation/pre_lowering/run_prelowering_batch_v5.sh" > "$LOGS/fig09_prelowering.log" 2>&1
    note "STAGE fig09 rc=$?"
    mkdir -p "$EV/fig09_prelowering"
    cp -a "$BENCH/ablation/pre_lowering/logs/batch_meta.txt" "$EV/fig09_prelowering/"
    for c in heat_s1_A heat_s1_B heat_s3_A heat_s3_B advdiff_A advdiff_B star_A star_B; do
      cp -a "$BENCH/ablation/pre_lowering/logs/$c.log" "$BENCH/ablation/pre_lowering/logs/$c.samples.csv" "$EV/fig09_prelowering/" 2>/dev/null
    done
    for j in heat_scale1 heat_scale3 advdiff_scale1 star_recur; do
      cp -a "$BENCH/ablation/pre_lowering/logs/cells/${j}_armA.json" "$BENCH/ablation/pre_lowering/logs/cells/${j}_armB.json" "$EV/fig09_prelowering/" 2>/dev/null
    done
  fi
  if [ "${SKIP_ROSTER:-0}" != "1" ]; then
    note "STAGE fig10 SICM roster + section 2"
    FROZEN_LOAD_GATE=8 FROZEN_START_EXT_CPU=700 FROZEN_EXT_CPU_LIMIT=1000 FROZEN_MAX_RETRIES=60 \
    FROZEN_CELLS="${ROSTER_CELLS:-
car_spatial_scotland:sicm/car_spatial:0.05:10
pgls:sicm/pgls:0.05:10
lmm_radon:sicm/linear_mixed_model:0.05:10
lmm_radon_n500:sicm/linear_mixed_model:0.05:10
lmm_radon_n919:sicm/linear_mixed_model:0.05:10
diamonds_n53920:sicm/diamonds_regression:0.05:10
hierarchical_mvn_K120_J2:sicm/hierarchical_mvn:0.01:10
phylo_pois:motivating/phylo_pois:0.1:10
car_spatial_nyc_n1921:sicm/car_spatial:0.05:10
}" bash "$BENCH/ablation/sicm_roster/run_frozen_batch.sh" "$CAMPAIGN" > "$LOGS/fig10_roster.log" 2>&1
    note "STAGE fig10 rc=$?"
  fi
  if [ "${SKIP_SVI_IR:-0}" != "1" ]; then
    note "STAGE SICM-on-VI x3 and section-2 IR"
    CAMPAIGN="$CAMPAIGN" CAMPAIGN_EV="$EV" IR_HLO_DIR="$EV/section2_phylo/hlo" STAGES="svi ir" \
      bash "$BENCH/run_ablation_uncapped.sh" > "$LOGS/svi_ir.log" 2>&1
    note "STAGE svi_ir rc=$?"
  fi
  note "uncapped phase done"
  ;;
pinned)
  note "STAGE pinned: fig11 impulse (core 8), fig12 impulse (32 40 48), fig13 (56), coal (64)"
  ( ONLY_PPL=impulse LOGDIR="$LOGS/fig11" CORES_LIST="0 8 16 24" MAX_ROW_RETRIES=40 \
      bash "$BENCH/run_cross_framework_singlethread.sh" > "$LOGS/fig11.driver.out" 2>&1
    note "  fig11 rc=$?" ) &
  ( VARIANT_FILTER='^ladder_impulse_' CORES="32 40 48" \
      bash "$BENCH/run_slicstan_per_variant.sh" "$LOGS/fig12" > "$LOGS/fig12.driver.out" 2>&1
    note "  fig12 rc=$?" ) &
  ( mkdir -p "$EV/fig13_generality/scaledcov_n200"
    EV200="$EV/fig13_generality/scaledcov_n200" \
      bash "$BENCH/run_generality_impulse_symmetric.sh" 56 "$LOGS/fig13" > "$LOGS/fig13.driver.out" 2>&1
    note "  fig13 rc=$?" ) &
  ( LOAD_GATE=24 START_EXT_CPU=1000 LOAD_MAX=24 EXT_CPU_LIMIT=1000 ABORT_SAMPLES=24 MAX_RETRIES=200 \
      "$BENCH/refreeze_cell.sh" coal_marginalized "$LOGS/coal" 2h none -- \
      "PPLBENCH_TASKSET_CPUS=64 ./run_pplbench.sh --threads 1 --ppls impulse standard/coal_mining_matched_init" \
      > "$LOGS/coal.driver.out" 2>&1
    note "  coal rc=$?" ) &
  wait
  note "pinned phase done"
  ;;
*) note "!! unknown phase $ph" ;;
esac
done
check_md5
note "reproduction done: $PHASES"
