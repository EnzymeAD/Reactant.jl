#!/bin/bash
# Post-pinned work, decoupled from the w1000 row (orchestrator, 2026-09-09 13:25):
# (1) the repetition-bimodality diagnostic, (2) the symmetric re-measurement of
# the Impulse generality cells with the XLA Eigen pool disabled. Both are pinned
# single-core cells on CCD 6 (core 48); the only other pinned work, the w1000
# row, runs on cores 64-88, so they do not share a die.
set -uo pipefail
B="/mnt4/sbrantq/probprog/oopsla/Reactant.jl/benchmark/probprog"
L="$B/logs/refreeze_2026-09-08"
cd "$B" || exit 1
echo "[$(date '+%F %T')] starting the post-pinned phase (decoupled from the w1000 row)"
./run_bimodality_diagnostic.sh "$L/bimodality" > "$L/bimodality.driver.log" 2>&1
echo "[$(date '+%F %T')] bimodality diagnostic rc=$?"
if [ -d "$L/fig13" ] && [ ! -d "$L/fig13_impulse_no_eigen_flag" ]; then
  mv "$L/fig13" "$L/fig13_impulse_no_eigen_flag"; echo "[$(date '+%F %T')] moved fig13 -> fig13_impulse_no_eigen_flag"
fi
./run_generality_impulse_symmetric.sh 48 "$L/fig13" > "$L/fig13_symmetric.driver.log" 2>&1
echo "[$(date '+%F %T')] symmetric Impulse generality pass rc=$?"
echo "[$(date '+%F %T')] post-pinned phase finished"
