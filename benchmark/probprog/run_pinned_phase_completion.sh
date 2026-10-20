#!/bin/bash
# Owns the marker that releases the post-pinned phase.
#
# run_pinned_after_uncapped.sh waits only on the jobs it spawned itself, and the
# SlicStan comparison no longer runs as two of those jobs: it is measured here as
# seven per-variant cells (run_slicstan_per_variant.sh). So the chain script's
# own "pinned phase finished" line now fires as soon as the two cross-framework
# sweeps end, which is earlier than the true end of the pinned phase.
#
# This script therefore runs the seven variant cells, waits for the chain
# script's line as well, and only when BOTH hold writes
# "pinned phase finished (ladder split)", which is what run_post_pinned.sh keys
# on. The marker lives in its own file because the chain script holds
# pinned_phase.log open on a truncating redirect, so a concurrent append to that
# file could be overwritten by the chain's next write.
set -uo pipefail
B="/mnt4/sbrantq/probprog/oopsla/Reactant.jl/benchmark/probprog"
L="$B/logs/refreeze_2026-09-08"
cd "$B" || exit 1

echo "[$(date '+%F %T')] starting the seven SlicStan per-variant cells"
bash ./run_slicstan_per_variant.sh "$L/fig12" > "$L/fig12/variants.log" 2>&1
echo "[$(date '+%F %T')] SlicStan per-variant pass rc=$?"

until grep -q "] pinned phase finished" "$L/pinned_phase.log" 2>/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] the chain script's cross-framework sweeps are finished too"

echo "[$(date '+%F %T')] pinned phase finished (ladder split)" >> "$L/pinned_phase_ladder_split.log"
echo "[$(date '+%F %T')] wrote the ladder-split completion marker"
