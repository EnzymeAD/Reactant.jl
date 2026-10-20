#!/bin/bash
# Re-measure the two cross-framework rows that were accepted only under the
# SUSTAINED load rule, now that the external build has ended and the machine is
# at its quiet floor, so the figure can use strict-clean cells everywhere.
#
# The rows, and why they are here (from EV/CELL_LEDGER.md):
#   robust        nuts_robust_regression_turing: external CPU instantaneous max
#                 5160%, five-minute mean 430%, 1.5% of samples over the limit
#   svi_logistic  all four cells: instantaneous max 1043%, five-minute mean
#                 about 330%, 1.8% of samples over the limit
# Every other cell of those two rows was already clean under both rules, but a
# row is measured as a unit (all four systems concurrently, sharing ambient
# conditions), so the whole row is re-run, never a single cell.
#
# NOTHING IS RELAXED: same driver, same gate environment as the sweep
# (LOAD_GATE=20 START_EXT_CPU=1000 LOAD_MAX=24 EXT_CPU_LIMIT=1000
# ABORT_SAMPLES=24), same cores 0/8/16/24, same one-replicate-per-cell rule.
# Note that this driver's own accept test is the sustained one, so strictness is
# decided afterwards by oopsla/tools/refreeze_cell_ledger.py, which prints both
# verdicts. MAX_ROW_RETRIES=2: if a row cannot land strictly clean within two
# attempts, the sustained-only cell it replaces STAYS the figure cell and the
# ledger says so.
#
# Runs into its own log directory, so the original sweep's cells are untouched
# and both measurements survive side by side in the evidence tree.
set -uo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
cd "$B" || exit 1
L="$B/logs/refreeze_2026-09-08"
OUT="${1:-$L/fig11_strict}"
WAIT_PID="${WAIT_FOR_PID:-2687606}"

# The rows re-use cores 0/8/16/24, which the main sweep holds until its last row
# (noisy_or) is done. Never overlap with it.
if [ -n "$WAIT_PID" ] && [ "$WAIT_PID" != "none" ]; then
  echo "[$(date '+%F %T')] waiting for the fig11 sweep (pid $WAIT_PID) to release cores 0/8/16/24"
  while ps -o args= -p "$WAIT_PID" 2>/dev/null | grep -q run_cross_framework_singlethread.sh; do
    sleep 60
  done
  echo "[$(date '+%F %T')] fig11 sweep finished; cores free"
fi

mkdir -p "$OUT/cells"
echo "[$(date '+%F %T')] strict re-measurement of rows: robust svi_logistic -> $OUT"
env LOAD_GATE=20 START_EXT_CPU=1000 LOAD_MAX=24 EXT_CPU_LIMIT=1000 ABORT_SAMPLES=24 \
    MAX_ROW_RETRIES=2 ROWS="robust svi_logistic" LOGDIR="$OUT" \
    ./run_cross_framework_singlethread.sh > "$OUT/nohup.log" 2>&1
echo "[$(date '+%F %T')] strict re-measurement rc=$?"
python3 /mnt4/sbrantq/probprog/oopsla/tools/refreeze_cell_ledger.py > "$OUT/ledger_after.txt" 2>&1
echo "[$(date '+%F %T')] strict rows pass finished"
