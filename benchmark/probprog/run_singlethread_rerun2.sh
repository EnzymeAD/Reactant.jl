#!/bin/bash
#
# Single-threaded cross-system rerun, v2: row-concurrent.
#
# Same protocol as run_singlethread_rerun.sh (--threads 1, taskset-pinned), but
# cells that are compared against each other (the systems of one model row) run
# CONCURRENTLY, each pinned to its own physical core on a distinct CCD, so all
# cells of a row share identical ambient machine conditions; rows run
# sequentially. Cores are drawn from distinct CCDs and avoid CCD 7 (cores
# 56-63), which the noisy-or agent session is using.
#
# The prior serial run (logs/rerun_singlethread_2026-08-24_0218) completed the
# phase-0 validation gate: PASSED (validate_nuts_results.json, copied here);
# that is REUSED (numerics-only, threading-independent). Its logistic row is
# NOT reused: a bug (nproc honoring OMP_NUM_THREADS) meant its taskset never
# applied, so that row ran with env caps but no pin; logistic is re-measured
# here under the fixed pin.
#
# Phases:
#   1. NUTS matched-init rows: robust_regression, n_schools,
#      crowd_sourced_annotation, noisy_or_topic (4 systems concurrent per row)
#   2. Impulse NUTS replicates 2,3 for all 5 models (10 concurrent runs)
#   3. SVI rows (4 systems concurrent per row) + Impulse SVI replicates
#   4. SlicStan ladder Impulse rungs (single job, internally serial)

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

STAMP="$(date +%Y-%m-%d_%H%M)"
LOGDIR="$SCRIPT_DIR/logs/rerun_singlethread_${STAMP}_concurrent"
PREV="$SCRIPT_DIR/logs/rerun_singlethread_2026-08-24_0218"
mkdir -p "$LOGDIR"
DRIVER_LOG="$LOGDIR/driver.log"
SO="$SCRIPT_DIR/../../deps/ReactantExtra/bazel-bin/libReactantExtra.so"

cp "$PREV/validate_nuts_results.json" "$LOGDIR/" 2>/dev/null
echo "$PREV" > "$LOGDIR/reused_from_prior_run.txt"
echo "phase0 validate reused from prior serial run (logistic row re-measured here: prior run's taskset never applied due to the nproc/OMP_NUM_THREADS bug)" >> "$LOGDIR/reused_from_prior_run.txt"

note() { echo "[$(date '+%F %T')] $*" | tee -a "$DRIVER_LOG"; }
so_md5() { md5sum "$SO" 2>/dev/null | awk '{print $1}'; }

provenance() {
    {
        echo "=== provenance @ $1 ==="
        date '+%F %T'
        echo "nproc=$(nproc) load=$(cut -d' ' -f1-3 /proc/loadavg)"
        echo "libReactantExtra.so md5=$(so_md5) mtime=$(stat -c %y "$SO" 2>/dev/null)"
        for r in Enzyme Enzyme-JAX Reactant.jl; do
            echo "$r head=$(git -C "$SCRIPT_DIR/../../../$r" rev-parse --short HEAD 2>/dev/null) dirty_tracked=$(git -C "$SCRIPT_DIR/../../../$r" status --porcelain 2>/dev/null | grep -cv '^??')"
        done
    } >> "$LOGDIR/provenance.txt"
}

BASE_MD5="$(so_md5)"
check_binary() {
    cur="$(so_md5)"
    if [ "$cur" != "$BASE_MD5" ]; then
        note "WARNING: libReactantExtra.so md5 CHANGED mid-sweep ($BASE_MD5 -> $cur) at $1"
    fi
}

# Physical cores on distinct CCDs (8 cores/CCD); CCD 7 (56-63) reserved for the
# noisy-or agent session.
CORES=(0 8 16 24 32 40 48 64 72 80)

declare -A PIDS=()
spawn() {
    # spawn <label> <slot> <timeout> <run_pplbench args...>
    local label="$1" slot="$2" tmo="$3"; shift 3
    local core="${CORES[$slot]}"
    note "SPAWN $label (core $core): run_pplbench.sh $*"
    PPLBENCH_TASKSET_CPUS="$core" timeout "$tmo" ./run_pplbench.sh "$@" \
        > "$LOGDIR/${label}.log" 2>&1 &
    PIDS[$!]="$label"
    # Stagger: pplbench output dirs are second-resolution timestamps.
    sleep 3
}

wait_batch() {
    local pid rc
    for pid in "${!PIDS[@]}"; do
        if wait "$pid"; then
            note "END   ${PIDS[$pid]}: OK"
        else
            rc=$?
            note "END   ${PIDS[$pid]}: FAILED (rc=$rc) see $LOGDIR/${PIDS[$pid]}.log"
        fi
    done
    PIDS=()
    check_binary "$1"
}

note "Concurrent single-threaded rerun starting; logs in $LOGDIR"
note "Reusing from $PREV: phase-0 validate (PASSED) and nuts_logistic_regression"
provenance "start"

# ---------------------------------------------------------------- Phase 1
note "Phase 1: NUTS matched-init rows, 4 systems concurrent per row"
for m in logistic_regression robust_regression n_schools crowd_sourced_annotation noisy_or_topic; do
    note "-- row: $m"
    spawn "nuts_${m}_numpyro" 0 6h --threads 1 --ppls numpyro "standard/${m}_matched_init"
    spawn "nuts_${m}_impulse" 1 6h --threads 1 --ppls impulse "standard/${m}_matched_init"
    spawn "nuts_${m}_turing"  2 6h --threads 1 --ppls turing  "standard/${m}_matched_init"
    spawn "nuts_${m}_stan"    3 6h --threads 1 --ppls stan    "standard/${m}_matched_init"
    wait_batch "nuts_${m}"
done
provenance "after-phase1"

# ---------------------------------------------------------------- Phase 2
note "Phase 2: Impulse NUTS replicates 2,3 (10 concurrent runs)"
slot=0
for rep in 2 3; do
    for m in logistic_regression robust_regression n_schools crowd_sourced_annotation noisy_or_topic; do
        spawn "nuts_${m}_impulse_rep${rep}" "$slot" 3h --threads 1 --ppls impulse "standard/${m}_matched_init"
        slot=$((slot + 1))
    done
done
wait_batch "phase2"
provenance "after-phase2"

# ---------------------------------------------------------------- Phase 3
note "Phase 3: SVI rows, systems concurrent per row"
note "-- row: svi logistic (plateau budget; Stan via its raised-budget config)"
spawn "svi_logistic_numpyro" 0 4h --threads 1 --ppls numpyro "svi/logistic_regression_plateau_budget"
spawn "svi_logistic_impulse" 1 4h --threads 1 --ppls impulse "svi/logistic_regression_plateau_budget"
spawn "svi_logistic_turing"  2 4h --threads 1 --ppls turing  "svi/logistic_regression_plateau_budget"
spawn "svi_logistic_stan"    3 4h --threads 1 --ppls stan    "svi/logistic_regression_matched_init"
wait_batch "svi_logistic"
for m in robust_regression n_schools; do
    note "-- row: svi $m"
    spawn "svi_${m}_numpyro" 0 3h --threads 1 --ppls numpyro "svi/${m}"
    spawn "svi_${m}_impulse" 1 3h --threads 1 --ppls impulse "svi/${m}"
    spawn "svi_${m}_turing"  2 3h --threads 1 --ppls turing  "svi/${m}"
    spawn "svi_${m}_stan"    3 3h --threads 1 --ppls stan    "svi/${m}"
    wait_batch "svi_${m}"
done
note "-- Impulse SVI replicates 2,3 (6 concurrent runs)"
slot=0
for rep in 2 3; do
    spawn "svi_logistic_impulse_rep${rep}" "$slot" 2h --threads 1 --ppls impulse "svi/logistic_regression_plateau_budget"; slot=$((slot + 1))
    spawn "svi_robust_impulse_rep${rep}"   "$slot" 2h --threads 1 --ppls impulse "svi/robust_regression"; slot=$((slot + 1))
    spawn "svi_n_schools_impulse_rep${rep}" "$slot" 2h --threads 1 --ppls impulse "svi/n_schools"; slot=$((slot + 1))
done
wait_batch "phase3-replicates"
provenance "after-phase3"

# ---------------------------------------------------------------- Phase 4
note "Phase 4: SlicStan ladder Impulse rungs (single job, internally serial)"
spawn "ladder_impulse" 0 6h --threads 1 --ppls impulse "related_work/gp_pois_regr_ladder"
wait_batch "ladder"
provenance "end"

note "Sweep complete. Logs: $LOGDIR (plus reused results in $PREV)"
