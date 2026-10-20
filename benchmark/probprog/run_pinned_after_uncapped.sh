#!/bin/bash
# Start the pinned phase as soon as the uncapped ablation phase is finished.
# The two phases must not overlap: the uncapped batches require the machine to
# themselves. Cores: fig11 rows on CCD 0-3, the noisy-or w1000 row on CCD 8-11,
# the two SlicStan cells on CCD 4 and 5, the generality pass on CCD 6.
set -uo pipefail
B="/mnt4/sbrantq/probprog/oopsla/Reactant.jl/benchmark/probprog"
L="$B/logs/refreeze_2026-09-08"
cd "$B" || exit 1
until grep -q "uncapped phase done" "$L/uncapped/driver.log" 2>/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] uncapped phase finished"

# Before the pinned phase (which is the only thing allowed to run concurrently
# with itself), finish the uncapped work: for every SICM-roster row whose two
# arms decorrelate before the end of the 20-draw window, re-measure over the
# matched prefix so the ledger carries BOTH ratios, the full-window one and the
# one whose whole window is matched. Rows that match over all 20 draws are
# skipped. This is an uncapped batch and runs alone, like the ones before it.
echo "[$(date '+%F %T')] matched-prefix re-measurement for the SICM roster"
FROZEN_LOAD_GATE="${FROZEN_LOAD_GATE:-12}" FROZEN_START_EXT_CPU="${FROZEN_START_EXT_CPU:-900}" FROZEN_EXT_CPU_LIMIT="${FROZEN_EXT_CPU_LIMIT:-1000}" FROZEN_MAX_RETRIES="${FROZEN_MAX_RETRIES:-400}" FROZEN_QUIET_HOLD="${FROZEN_QUIET_HOLD:-15}" FROZEN_ABORT_SAMPLES="${FROZEN_ABORT_SAMPLES:-1}"   bash "$B/ablation/sicm_roster/run_prefix_window.sh" refreeze_2026-09-08   > "$L/uncapped/prefix_window.log" 2>&1
echo "[$(date '+%F %T')] matched-prefix re-measurement rc=$?; starting the pinned phase"
mkdir -p "$L"/{fig11/cells,fig11_w1000/cells,fig12/cells,fig13/out/scaledcov,fig13/out/gen}
COMMON="LOAD_GATE=20 START_EXT_CPU=1000 LOAD_MAX=24 EXT_CPU_LIMIT=1000 ABORT_SAMPLES=24"
env $COMMON MAX_ROW_RETRIES=200 \
  ROWS="logistic robust n_schools svi_logistic svi_robust svi_n_schools annotation noisy_or" \
  ./run_cross_framework_singlethread.sh > "$L/fig11/nohup.log" 2>&1 &
env $COMMON MAX_ROW_RETRIES=200 ROWS="noisy_or_w1000" CORES_LIST="64 72 80 88" \
  LOGDIR="$L/fig11_w1000" \
  ./run_cross_framework_singlethread.sh > "$L/fig11_w1000/nohup.log" 2>&1 &
env $COMMON MAX_RETRIES=200 ./refreeze_cell.sh ladder_stan "$L/fig12" 8h none -- \
  "PPLBENCH_TASKSET_CPUS=32 ./run_pplbench.sh --threads 1 --ppls stan related_work/gp_pois_regr_ladder_4trials" \
  > "$L/fig12/ladder_stan.driver.log" 2>&1 &
env $COMMON MAX_RETRIES=200 ./refreeze_cell.sh ladder_impulse "$L/fig12" 6h none -- \
  "PPLBENCH_TASKSET_CPUS=40 ./run_pplbench.sh --threads 1 --ppls impulse related_work/gp_pois_regr_ladder_4trials" \
  > "$L/fig12/ladder_impulse.driver.log" 2>&1 &
env $COMMON MAX_RETRIES=200 ./run_generality_pinned.sh 48 > "$L/fig13.driver.log" 2>&1 &
wait
echo "[$(date '+%F %T')] pinned phase finished"
