#!/bin/bash
# Second half of the 2026-09-10 problem-size sweep. Waits for the first driver
# to exit, then runs the remaining cells under the same gates. Separate file
# because editing a running bash script is unsafe (bash re-reads it by byte
# offset).
#
# Sizes here were deferred until the n=500 / D=500 points made their cost
# measurable rather than guessed: the measured scaling of both configurations
# is far below the flop count, so n=1000, n=1921 and D=1000 are affordable.
set -uo pipefail
BENCH="/mnt4/sbrantq/probprog/oopsla/Reactant.jl/benchmark/probprog"
LOGS="$BENCH/logs/size_sweep_2026-09-10"
cd "$BENCH" || exit 1
DRV="$LOGS/supervisor.log"
note() { echo "[$(date '+%F %T')] $*" | tee -a "$DRV"; }

while pgrep -u "$(id -un)" -f run_size_sweep.sh > /dev/null; do sleep 60; done
note "tail driver start; first driver has exited"

export FROZEN_LOAD_GATE=8 FROZEN_START_EXT_CPU=700 FROZEN_EXT_CPU_LIMIT=1000 FROZEN_MAX_RETRIES=60
export PRELOWER_LOAD_GATE=8 PRELOWER_START_EXT_CPU=700 PRELOWER_EXT_CPU_LIMIT=1000 PRELOWER_MAX_RETRIES=60
TAG="size_sweep_2026-09-10"

roster () { FROZEN_CELLS="$2" bash "$BENCH/ablation/sicm_roster/run_frozen_batch.sh" "$TAG" > "$LOGS/$1.log" 2>&1; note "STAGE $1 rc=$?"; }

for st in ${STAGES:-radon919 roster_d roster_e prelower_c}; do
case "$st" in
radon919)
  note "STAGE radon919: retry Radon LMM n=919 (its first warmup segfaulted)"
  roster radon919 "
lmm_radon_n919:sicm/linear_mixed_model:0.05:10
" ;;
roster_d)
  note "STAGE roster_d: NYC traffic CAR n=1000"
  roster roster_d "
car_spatial_nyc_n1000:sicm/car_spatial:0.05:10
" ;;
roster_e)
  note "STAGE roster_e: NYC traffic CAR n=1921 (whole map)"
  roster roster_e "
car_spatial_nyc_n1921:sicm/car_spatial:0.05:10
" ;;
prelower_c)
  note "STAGE prelower_c: advection-diffusion D=1000 T=500"
  PRELOWER_CELLS="advdiff_s4_w advdiff_s4" \
    bash "$BENCH/ablation/pre_lowering/run_prelowering_batch_v5.sh" \
    > "$LOGS/prelower_c.log" 2>&1
  note "STAGE prelower_c rc=$?"
  cp "$BENCH/ablation/pre_lowering/logs/batch_meta.txt" "$LOGS/prelower_c.batch_meta.txt" 2>/dev/null ;;
*) note "!! unknown stage $st" ;;
esac
done
note "size sweep stages done: tail"
