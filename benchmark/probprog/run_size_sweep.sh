#!/bin/bash
# Problem-size sweep for the two ablation figures, 2026-09-10.
#
# Same binary and same protocol as the 2026-09-08 re-freeze: one shared adapted
# state per (row, size) minted by a single adaptation-ON warmup, both
# configurations then sampling with adaptation OFF from that state with matched
# seeds, median of 5 in-process executions of 20 draws, compilation excluded,
# uncapped regime, one cell at a time, alone.
#
# Gates are the re-freeze's uncapped ones verbatim (REFREEZE_2026-09-08.md
# D-4, D-5, D-6): start at procs_running < 8 and external CPU < 700% held for
# 30 s, strict INSTANTANEOUS acceptance at external CPU < 1000%, watchdog
# aborts after 2 min over the limit and the cell is retried, never gate on raw
# load1 in the uncapped regime. This script only sequences the two campaign
# drivers, which own the gates.
set -uo pipefail
BENCH="/mnt4/sbrantq/probprog/oopsla/Reactant.jl/benchmark/probprog"
LOGS="$BENCH/logs/size_sweep_2026-09-10"
cd "$BENCH" || exit 1
mkdir -p "$LOGS"
DRV="$LOGS/supervisor.log"
note() { echo "[$(date '+%F %T')] $*" | tee -a "$DRV"; }

export FROZEN_LOAD_GATE="${FROZEN_LOAD_GATE:-8}"
export FROZEN_START_EXT_CPU="${FROZEN_START_EXT_CPU:-700}"
export FROZEN_EXT_CPU_LIMIT="${FROZEN_EXT_CPU_LIMIT:-1000}"
export FROZEN_MAX_RETRIES="${FROZEN_MAX_RETRIES:-60}"
export PRELOWER_LOAD_GATE="${PRELOWER_LOAD_GATE:-8}"
export PRELOWER_START_EXT_CPU="${PRELOWER_START_EXT_CPU:-700}"
export PRELOWER_EXT_CPU_LIMIT="${PRELOWER_EXT_CPU_LIMIT:-1000}"
export PRELOWER_MAX_RETRIES="${PRELOWER_MAX_RETRIES:-60}"

TAG="size_sweep_2026-09-10"
STAGES="${STAGES:-roster_a roster_b roster_c prelower_a prelower_b}"
note "size sweep start; stages: $STAGES"
cp -n "$BENCH/ablation/pre_lowering/logs/batch_meta.txt" \
      "$LOGS/prelowering_batch_meta_before_sweep.txt" 2>/dev/null
note "md5: $(md5sum /mnt4/sbrantq/probprog/oopsla/Reactant.jl/deps/ReactantExtra/bazel-bin/libReactantExtra.so | cut -d' ' -f1)"

roster () {  # $1 = stage log name, $2 = newline-separated cell list
  FROZEN_CELLS="$2" bash "$BENCH/ablation/sicm_roster/run_frozen_batch.sh" "$TAG" \
    > "$LOGS/$1.log" 2>&1
  note "STAGE $1 rc=$?"
}
# The pre-lowering driver keeps its DEFAULT log dir on purpose: the harness
# writes its per-cell JSON (samples, accept/diverge flags, IR counts) to
# ablation/pre_lowering/logs/cells unconditionally, and aggregate_v2.py reads
# the cell JSON from <logdir>/cells, so a moved logdir would silently drop the
# arms-equivalence gate. New cells carry new file names (heat_scale6/7,
# advdiff_scale3/4) and collide with nothing; only batch_meta.txt is rewritten,
# and the re-freeze copy of it is preserved under
# oopsla/evidence/refreeze_2026-09-08/fig09_prelowering/batch_meta.txt and
# copied into this sweep's log dir before the first stage.
prelower () {  # $1 = stage log name, $2 = space-separated cell list
  PRELOWER_CELLS="$2" bash "$BENCH/ablation/pre_lowering/run_prelowering_batch_v5.sh" \
    > "$LOGS/$1.log" 2>&1
  note "STAGE $1 rc=$?"
  cp "$BENCH/ablation/pre_lowering/logs/batch_meta.txt" "$LOGS/$1.batch_meta.txt" 2>/dev/null
}

for st in $STAGES; do
case "$st" in
roster_a)
  note "STAGE roster_a: diamonds N=20000/53920, radon n=500/919, hier MVN K=60 J=5"
  roster roster_a "
diamonds_n20000:sicm/diamonds_regression:0.05:10
diamonds_n53920:sicm/diamonds_regression:0.05:10
lmm_radon_n500:sicm/linear_mixed_model:0.05:10
lmm_radon_n919:sicm/linear_mixed_model:0.05:10
hierarchical_mvn_K60_J5:sicm/hierarchical_mvn:0.01:10
" ;;
roster_b)
  note "STAGE roster_b: hier MVN K=120 J=2, K=240 J=2"
  roster roster_b "
hierarchical_mvn_K120_J2:sicm/hierarchical_mvn:0.01:10
hierarchical_mvn_K240_J2:sicm/hierarchical_mvn:0.01:10
" ;;
roster_c)
  note "STAGE roster_c: NYC traffic CAR n=500"
  roster roster_c "
car_spatial_nyc_n500:sicm/car_spatial:0.05:10
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
prelower_a)
  note "STAGE prelower_a: heat D=300 T=400, advection-diffusion D=500 T=250"
  prelower prelower_a "heat_s7_w heat_s7 advdiff_s3_w advdiff_s3" ;;
prelower_b)
  note "STAGE prelower_b: heat D=1000 T=100"
  prelower prelower_b "heat_s6_w heat_s6" ;;
prelower_c)
  note "STAGE prelower_c: advection-diffusion D=1000 T=500"
  prelower prelower_c "advdiff_s4_w advdiff_s4" ;;
*) note "!! unknown stage $st" ;;
esac
done
note "size sweep stages done: $STAGES"
