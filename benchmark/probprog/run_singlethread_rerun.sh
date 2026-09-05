#!/bin/bash
#
# Full single-threaded cross-system rerun (protocol decision 2026-08-24):
# every cross-system CPU comparison runs --threads 1, enforced by taskset in
# run_pplbench.sh. Scope per paper/CaseStudies.tex section 5.1 draft bullets:
#   Phase 0: adapt-off NUTS validation gate (no-regression; ABORTS sweep on fail)
#   Phase 1: NUTS matched-init, 5 models x {numpyro,impulse,turing,stan}
#   Phase 2: Impulse NUTS replicates 2 and 3 per model
#   Phase 3: SVI plateau-budget suite (+ Impulse replicates)
#   Phase 4: SlicStan ladder, Impulse rungs only (Stan rungs stand: they were
#            already de facto single-threaded)
# NOT in scope: ablations (run uncapped, --threads all), Comrade (GPU),
# per-move MH suite (separate harnesses; handled outside this driver).
#
# Usage: ./run_singlethread_rerun.sh   (intended to run unattended; ~8-12h)

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

STAMP="$(date +%Y-%m-%d_%H%M)"
LOGDIR="$SCRIPT_DIR/logs/rerun_singlethread_${STAMP}"
mkdir -p "$LOGDIR"
DRIVER_LOG="$LOGDIR/driver.log"
SO="$SCRIPT_DIR/../../deps/ReactantExtra/bazel-bin/libReactantExtra.so"

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
        note "WARNING: libReactantExtra.so md5 CHANGED mid-sweep ($BASE_MD5 -> $cur) at $1; Impulse cells are no longer single-build"
    fi
}

run_cfg() {
    # run_cfg <label> <timeout> <run_pplbench args...>
    local label="$1"; shift
    local tmo="$1"; shift
    note "START $label: run_pplbench.sh $*"
    if timeout "$tmo" ./run_pplbench.sh "$@" > "$LOGDIR/${label}.log" 2>&1; then
        note "END   $label: OK"
    else
        note "END   $label: FAILED (rc=$?) see $LOGDIR/${label}.log"
    fi
    check_binary "$label"
}

note "Single-threaded rerun starting; logs in $LOGDIR"
provenance "start"

# ---------------------------------------------------------------- Phase 0
note "Phase 0: adapt-off NUTS validation gate (THREADS=1)"
if THREADS=1 VALIDATE_TIMEOUT=3600 \
   VALIDATE_LOG_DIR="$LOGDIR/validate_transcripts" \
   VALIDATE_RESULTS_JSON="$LOGDIR/validate_nuts_results.json" \
   ./run_all_validate.sh > "$LOGDIR/phase0_validate.log" 2>&1; then
    note "Phase 0: validation gate PASSED"
else
    note "Phase 0: validation gate FAILED; ABORTING sweep (see $LOGDIR/phase0_validate.log)"
    provenance "abort-after-phase0"
    exit 1
fi
check_binary "phase0"

NUTS_MODELS=(logistic_regression robust_regression n_schools crowd_sourced_annotation noisy_or_topic)
ALL4="numpyro,impulse,turing,stan"

# ---------------------------------------------------------------- Phase 1
note "Phase 1: NUTS matched-init, all four systems"
for m in "${NUTS_MODELS[@]}"; do
    run_cfg "nuts_${m}" 6h --threads 1 --ppls "$ALL4" "standard/${m}_matched_init"
done
provenance "after-phase1"

# ---------------------------------------------------------------- Phase 2
note "Phase 2: Impulse NUTS replicates 2 and 3"
for rep in 2 3; do
    for m in "${NUTS_MODELS[@]}"; do
        run_cfg "nuts_${m}_impulse_rep${rep}" 3h --threads 1 --ppls impulse "standard/${m}_matched_init"
    done
done
provenance "after-phase2"

# ---------------------------------------------------------------- Phase 3
note "Phase 3: SVI plateau-budget suite"
run_cfg "svi_logistic_plateau" 4h --threads 1 --ppls numpyro,impulse,turing "svi/logistic_regression_plateau_budget"
run_cfg "svi_logistic_stan"    2h --threads 1 --ppls stan "svi/logistic_regression_matched_init"
run_cfg "svi_robust"           3h --threads 1 --ppls "$ALL4" "svi/robust_regression"
run_cfg "svi_n_schools"        3h --threads 1 --ppls "$ALL4" "svi/n_schools"
for rep in 2 3; do
    run_cfg "svi_logistic_impulse_rep${rep}" 2h --threads 1 --ppls impulse "svi/logistic_regression_plateau_budget"
    run_cfg "svi_robust_impulse_rep${rep}"   2h --threads 1 --ppls impulse "svi/robust_regression"
    run_cfg "svi_n_schools_impulse_rep${rep}" 2h --threads 1 --ppls impulse "svi/n_schools"
done
provenance "after-phase3"

# ---------------------------------------------------------------- Phase 4
note "Phase 4: SlicStan ladder, Impulse rungs (3 rungs: no-pre-opt, std passes, SICM)"
run_cfg "ladder_impulse" 6h --threads 1 --ppls impulse "related_work/gp_pois_regr_ladder"
provenance "end"

note "Sweep complete. Logs: $LOGDIR"
