#!/bin/bash
# Generality figure, Gen-tutorial line-model rows: the line model of Gen's
# "Introduction to Modeling in Gen" under the block prior-resimulation move of
# "Iterative Inference in Gen" (mh on {slope, intercept} once per sweep, 500
# sweeps), data in linreg_tutorial_data.json (make_linreg_tutorial_data.jl).
#
# Same protocol as run_generality_coal_jarrett.sh: ONE pinned core, BLAS/OMP at
# one thread, the XLA Eigen pool disabled for every XLA system (Impulse and
# GenJAX), 20 timed in-process repetitions after one discarded warm-up, median
# per-move time taken by summarize_generality.py, every cell gated by
# refreeze_cell.sh with strict instantaneous acceptance.
#
# Cells, in order:
#   figure rows     impulse (SICM on and off in one process), Gen.jl, GenJAX,
#                   for row A (the tutorial's 11 points) and row B (N = 10000)
#   evidence only   impulse at N = 100 and N = 1000 (crossover curve)
#                   impulse observation-as-argument variant (IMPULSE_LT_OBS=arg)
#                   at A, B, N100, N1000 (attainable ratio once the observations
#                   are loop-invariant; not the paper's cells)
# Every Impulse cell writes its final HLO (both arms) to $IRDIR.
#
# Usage: run_generality_linreg_tutorial.sh [core] [logdir] [irdir]
set -uo pipefail
B="$(cd "$(dirname "$0")" && pwd)"; cd "$B" || exit 1
CORE="${1:-48}"
LOG="${2:-$B/logs/generality_linreg_tutorial}"
IRDIR="${3:-$LOG/ir}"
mkdir -p "$LOG/cells" "$IRDIR"
export STRICT="${STRICT:-1}"
DATA="$B/linreg_tutorial_data.json"
JULIA_ENV="CUDA_VISIBLE_DEVICES=\"\" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false"
GENJAX_PY=/home/sbrantq/anaconda3/envs/genjax-bench/bin/python
JAX_ENV="CUDA_VISIBLE_DEVICES=\"\" JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false"
G="LOAD_GATE=${LOAD_GATE:-12} START_EXT_CPU=${START_EXT_CPU:-900} LOAD_MAX=${LOAD_MAX:-24} \
EXT_CPU_LIMIT=${EXT_CPU_LIMIT:-1000} ABORT_SAMPLES=${ABORT_SAMPLES:-24} MAX_RETRIES=${MAX_RETRIES:-200}"
run() { echo "[$(date '+%F %T')] === $1"; env $G ./refreeze_cell.sh "$1" "$LOG" "$2" "$CORE" -- "$3"; }
imp() {  # imp <row> <obs>
  run "impulse_linreg_tutorial_${2}_$1" 120m \
    "$JULIA_ENV IMPULSE_LT_OBS=$2 IMPULSE_LT_IR_OUT=$IRDIR julia -t 1 --project=. impulse_mh_linreg_tutorial_harness.jl $DATA $1"
}

CELLS="${CELLS:-impulse_A impulse_B gen_A gen_B genjax_A genjax_B argobs_A argobs_B impulse_N100 impulse_N1000 argobs_N100 argobs_N1000}"
for c in $CELLS; do
  case "$c" in
    impulse_*) imp "${c#impulse_}" trace ;;
    argobs_*)  imp "${c#argobs_}" arg ;;
    gen_A|gen_B)       run "gen_linreg_tutorial_${c#gen_}" 60m "$JULIA_ENV julia -t 1 --project=gen_env gen_mh_linreg_tutorial_harness.jl $DATA ${c#gen_}" ;;
    genjax_A|genjax_B) run "genjax_linreg_tutorial_${c#genjax_}" 30m "$JAX_ENV $GENJAX_PY genjax_mh_linreg_tutorial_harness.py $DATA ${c#genjax_}" ;;
    *) echo "unknown cell $c"; exit 1 ;;
  esac
done
echo "[$(date '+%F %T')] linreg tutorial generality cells done; logs in $LOG"
