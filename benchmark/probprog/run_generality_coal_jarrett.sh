#!/bin/bash
# Generality figure, coal-mining change-point entry on the recorded Jarrett
# series (COAL_DATA=jarrett): the three programmable-MH cells (Impulse, Gen.jl
# vectorized, GenJAX), one at a time, on ONE pinned core.
#
# Same protocol as run_generality_pinned.sh / run_generality_impulse_symmetric.sh:
# taskset to one core, BLAS/OMP capped at one thread, the XLA Eigen pool
# disabled for every XLA system (Impulse and GenJAX), 20 timed in-process
# repetitions after one discarded warm-up repetition, median per-move time taken
# by summarize_generality.py, every cell gated by refreeze_cell.sh with the
# symmetric driver's gates and strict instantaneous acceptance.
#
# Each harness also prints, after its timed repetitions, the switch histogram
# of the timed chain and of one longer untimed validation chain
# (COAL_VAL_SWEEPS sweeps) with the two rate means; the validation does not
# enter the timings.
#
# Usage: run_generality_coal_jarrett.sh [core] [logdir]
set -uo pipefail
B="$(cd "$(dirname "$0")" && pwd)"; cd "$B" || exit 1
CORE="${1:-56}"
LOG="${2:-$B/logs/generality_coal_jarrett}"
mkdir -p "$LOG/cells"
export STRICT="${STRICT:-1}"
VAL="COAL_DATA=jarrett COAL_VAL_SWEEPS=${COAL_VAL_SWEEPS:-500000}"
JULIA_ENV="CUDA_VISIBLE_DEVICES=\"\" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false $VAL"
GENJAX_PY=/home/sbrantq/anaconda3/envs/genjax-bench/bin/python
JAX_ENV="CUDA_VISIBLE_DEVICES=\"\" JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false $VAL"
G="LOAD_GATE=${LOAD_GATE:-12} START_EXT_CPU=${START_EXT_CPU:-900} LOAD_MAX=${LOAD_MAX:-24} \
EXT_CPU_LIMIT=${EXT_CPU_LIMIT:-1000} ABORT_SAMPLES=${ABORT_SAMPLES:-24} MAX_RETRIES=${MAX_RETRIES:-200}"
run() { echo "[$(date '+%F %T')] === $1"; env $G ./refreeze_cell.sh "$1" "$LOG" "$2" "$CORE" -- "$3"; }

CELLS="${CELLS:-impulse_coal_jarrett gen_coal_jarrett genjax_coal_jarrett}"
for c in $CELLS; do
  case "$c" in
    impulse_coal_jarrett) run "$c" 40m "$JULIA_ENV julia -t 1 --project=. impulse_coal_mh_masked.jl" ;;
    gen_coal_jarrett)     run "$c" 30m "$JULIA_ENV julia -t 1 --project=gen_env gen_coal_mh_vec_harness.jl" ;;
    genjax_coal_jarrett)  run "$c" 30m "$JAX_ENV $GENJAX_PY genjax_coal_mh_harness.py" ;;
    *) echo "unknown cell $c"; exit 1 ;;
  esac
done
echo "[$(date '+%F %T')] coal jarrett generality cells done; logs in $LOG"
