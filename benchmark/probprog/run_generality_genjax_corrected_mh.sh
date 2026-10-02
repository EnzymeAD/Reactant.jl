#!/bin/bash
# Figure 13 GenJAX cells with the corrected MH acceptance (2026-09-29): the
# outlier, line-versus-sine and coal (Jarrett series) entries. GenJAX 0.10.3's
# Regenerate weight is the joint score difference; the harnesses now accept on
# the edit weight minus the projected score change of the selected sites (see
# oopsla/evidence/coal_jarrett_2026-09-29/GENJAX_REGENERATE_WEIGHT.md).
#
# Same protocol as run_generality_pinned.sh: one core, BLAS/OMP/MKL at one
# thread, the XLA Eigen pool disabled, 20 timed repetitions after a discarded
# warm-up, median taken by summarize_generality.py, refreeze_cell.sh strict
# gate. After its timed repetitions each cell runs the prior-invariance check of
# its own harness `step` on a prior-only trace (untimed), so the accepted cell
# carries the proof of the corrected kernel.
#
# Usage: run_generality_genjax_corrected_mh.sh [core] [logdir]
set -uo pipefail
B="$(cd "$(dirname "$0")" && pwd)"; cd "$B" || exit 1
CORE="${1:-56}"
LOG="${2:-$B/logs/generality_genjax_corrected_mh}"
mkdir -p "$LOG/cells"
export STRICT="${STRICT:-1}"
PY=/home/sbrantq/anaconda3/envs/genjax-bench/bin/python
PI=/mnt4/sbrantq/probprog/oopsla/evidence/coal_jarrett_2026-09-29/mh_prior_invariance/genjax_fig13_prior_invariance.py
JAX_ENV='CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false'
G="LOAD_GATE=${LOAD_GATE:-24} START_EXT_CPU=${START_EXT_CPU:-900} LOAD_MAX=${LOAD_MAX:-24} \
EXT_CPU_LIMIT=${EXT_CPU_LIMIT:-1000} ABORT_SAMPLES=${ABORT_SAMPLES:-24} MAX_RETRIES=${MAX_RETRIES:-200}"
run() { echo "[$(date '+%F %T')] === $1"; env $G ./refreeze_cell.sh "$1" "$LOG" "$2" "$CORE" -- "$3"; }
pi() { echo "$JAX_ENV PI_MODELS=$1 PI_KERNELS=fixed PI_SWEEPS=200000 PI_OUT=$LOG/prior_invariance_$1.json $PY $PI"; }

CELLS="${CELLS:-genjax_outlier genjax_linesine genjax_coal_jarrett}"
for c in $CELLS; do
  case "$c" in
    genjax_outlier)      run "$c" 40m "$JAX_ENV $PY genjax_outlier_mh_harness.py && $(pi outlier)" ;;
    genjax_linesine)     run "$c" 40m "$JAX_ENV $PY genjax_linesine_mh_harness.py && $(pi linesine)" ;;
    genjax_coal_jarrett) run "$c" 40m "$JAX_ENV COAL_DATA=jarrett COAL_VAL_SWEEPS=500000 $PY genjax_coal_mh_harness.py && $(pi coal)" ;;
    *) echo "unknown cell $c"; exit 1 ;;
  esac
done
echo "[$(date '+%F %T')] corrected GenJAX generality cells done; logs in $LOG"
