#!/bin/bash
# Figure 13 Impulse cells with the corrected MH acceptance (2026-09-29): the
# outlier, line-versus-sine and coal (Jarrett series) entries. The Impulse MH
# lowering now subtracts the Hastings term of the prior-regenerated sites
# (oopsla/evidence/mh_acceptance_fix_2026-09-29/README.md), so every Impulse MH
# program changed IR and is re-measured on the fixed binary.
#
# Same protocol as run_generality_impulse_symmetric.sh and
# run_generality_coal_jarrett.sh: taskset to one core, BLAS/OMP/MKL at one
# thread, the XLA Eigen pool disabled, 20 timed in-process repetitions after one
# discarded warm-up, median per-move time taken by summarize_generality.py,
# refreeze_cell.sh gate with strict instantaneous acceptance. The harnesses are
# unchanged. After its timed repetitions each cell runs, untimed, the validation
# script (joint density against a hand-computed value; the timed chains
# reproduced bit for bit by an accounting copy of the sweep body, giving their
# acceptance rates; prior invariance of the same sweep body on a prior-only
# model), so an accepted cell carries its validation.
#
# Usage: run_generality_impulse_corrected_mh.sh [core] [logdir]
set -uo pipefail
B="$(cd "$(dirname "$0")" && pwd)"; cd "$B" || exit 1
CORE="${1:-48}"
LOG="${2:-$B/logs/generality_impulse_corrected_mh}"
mkdir -p "$LOG/cells"
export STRICT="${STRICT:-1}"
VAL=/mnt4/sbrantq/probprog/oopsla/evidence/fig13_mh_retime_2026-09-29/scripts/impulse_fig13_mh_validation.jl
JULIA_ENV='CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false'
COAL="COAL_DATA=jarrett COAL_VAL_SWEEPS=${COAL_VAL_SWEEPS:-500000}"
G="LOAD_GATE=${LOAD_GATE:-12} START_EXT_CPU=${START_EXT_CPU:-900} LOAD_MAX=${LOAD_MAX:-24} \
EXT_CPU_LIMIT=${EXT_CPU_LIMIT:-1000} ABORT_SAMPLES=${ABORT_SAMPLES:-24} MAX_RETRIES=${MAX_RETRIES:-200}"
run() { echo "[$(date '+%F %T')] === $1"; env $G ./refreeze_cell.sh "$1" "$LOG" "$2" "$CORE" -- "$3"; }
val() { echo "$JULIA_ENV $2 VAL_MODEL=$1 PI_SWEEPS=${PI_SWEEPS:-200000} VAL_OUT=$LOG/validation_$1.json julia -t 1 --project=. $VAL"; }

CELLS="${CELLS:-impulse_outlier impulse_linesine impulse_coal_jarrett}"
for c in $CELLS; do
  case "$c" in
    impulse_outlier)      run "$c" 60m "$JULIA_ENV julia -t 1 --project=. impulse_outlier_mh_harness.jl && $(val outlier '')" ;;
    impulse_linesine)     run "$c" 60m "$JULIA_ENV julia -t 1 --project=. impulse_linesine_mh_harness.jl && $(val linesine '')" ;;
    impulse_coal_jarrett) run "$c" 60m "$JULIA_ENV $COAL julia -t 1 --project=. impulse_coal_mh_masked.jl && $(val coal "$COAL")" ;;
    *) echo "unknown cell $c"; exit 1 ;;
  esac
done
echo "[$(date '+%F %T')] corrected Impulse generality cells done; logs in $LOG"
