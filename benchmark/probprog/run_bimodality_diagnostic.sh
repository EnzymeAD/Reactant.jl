#!/bin/bash
# Diagnostic for the slow-repetition cluster in the generality cells.
#
# This is a LEDGER AND CAPTION diagnostic. It does not change the figure
# protocol and its cells are not figure cells.
#
# Hypothesis under test: under a one-core taskset, Reactant's XLA still starts
# its full CPU thread pool (run_pplbench.sh records that this XLA rejects
# --intra_op_parallelism_threads, so the pool is sized from sched_getaffinity at
# client creation), and the pool threads contend with the main thread for the
# single pinned core. If so, giving the cell a second hardware thread on the
# same physical core, or turning the pool off, should remove the slow cluster.
#
# Three configurations of the cheapest Impulse cell (linear regression, flat):
#   base    exactly as the figure measures it: one hardware thread, core 48
#   smt2    two hardware threads on the SAME physical core (48 and its sibling
#           144), so pool threads have somewhere to run without preempting the
#           main thread and without adding a second physical core
#   noeigen one hardware thread, plus XLA_FLAGS=--xla_cpu_multi_thread_eigen=false.
#           That flag IS present in this binary (has_/set_/clear_ symbols for
#           xla_cpu_multi_thread_eigen), and it is what the GenJAX cells of the
#           figure already run with, so this also removes an env asymmetry
#           between the two systems.
#
# Each configuration runs the harness unchanged (20 timed repetitions after a
# discarded warm-up) through refreeze_cell.sh, so each gets the campaign's
# quiet-window gate, load sampling and core-steal record.
set -uo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
cd "$B" || exit 1
LOG="${1:-$B/logs/refreeze_2026-09-08/bimodality}"
CORE="${DIAG_CORE:-48}"
SIB="$(cut -d, -f2 < /sys/devices/system/cpu/cpu${CORE}/topology/thread_siblings_list)"
mkdir -p "$LOG/cells"
CAPS='CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1'
G="LOAD_GATE=${LOAD_GATE:-12} START_EXT_CPU=${START_EXT_CPU:-900} LOAD_MAX=${LOAD_MAX:-24} \
EXT_CPU_LIMIT=${EXT_CPU_LIMIT:-1000} ABORT_SAMPLES=${ABORT_SAMPLES:-24} MAX_RETRIES=${MAX_RETRIES:-60}"

echo "[$(date '+%F %T')] bimodality diagnostic: core $CORE, SMT sibling $SIB"

env $G ./refreeze_cell.sh bimod_base "$LOG" 30m "$CORE" -- \
  "$CAPS IMPULSE_MH_ARM=flat julia -t 1 --project=. impulse_mh_linreg_harness.jl"

env $G ./refreeze_cell.sh bimod_smt2 "$LOG" 30m "${CORE},${SIB}" -- \
  "$CAPS IMPULSE_MH_ARM=flat julia -t 1 --project=. impulse_mh_linreg_harness.jl"

env $G ./refreeze_cell.sh bimod_noeigen "$LOG" 30m "$CORE" -- \
  "$CAPS XLA_FLAGS=--xla_cpu_multi_thread_eigen=false IMPULSE_MH_ARM=flat julia -t 1 --project=. impulse_mh_linreg_harness.jl"

echo "[$(date '+%F %T')] bimodality diagnostic done"
python3 /mnt/sbrantq/probprog/oopsla/tools/repetition_clusters.py "$LOG" \
  --json /mnt/sbrantq/probprog/oopsla/evidence/refreeze_2026-09-08/fig13_generality/bimodality/clusters_diagnostic.json \
  | tee /mnt/sbrantq/probprog/oopsla/evidence/refreeze_2026-09-08/fig13_generality/bimodality/clusters_diagnostic.txt
