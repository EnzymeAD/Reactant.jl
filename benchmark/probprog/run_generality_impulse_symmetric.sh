#!/bin/bash
# Re-measure the Impulse cells of the generality figure under the DECLARED
# single-core protocol, i.e. with the same XLA thread-pool setting every other
# system receives.
#
# run_pplbench.sh sets XLA_FLAGS=--xla_cpu_multi_thread_eigen=false for every
# framework at --threads 1. Until 2026-09-08 run_generality_pinned.sh applied it
# only to the GenJAX cells, so the Impulse and GenJAX cells of the same figure
# ran on different thread settings. These cells restore the symmetry; the Gen.jl
# and GenJAX cells are unaffected and are NOT re-run (Gen.jl has no XLA, and the
# GenJAX cells already had the flag).
#
# Same core, same 20 repetitions, same gates as the pass they replace.
set -uo pipefail
B="$(cd "$(dirname "$0")" && pwd)"; cd "$B" || exit 1
ROOT="$(cd "$B/../.." && pwd)"
CORE="${1:-48}"
LOG="${2:-$B/logs/refreeze_2026-09-08/fig13}"
mkdir -p "$LOG/cells" "$LOG/out/scaledcov" "$LOG/out/gen"
JULIA_ENV='CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false'
G="LOAD_GATE=${LOAD_GATE:-12} START_EXT_CPU=${START_EXT_CPU:-900} LOAD_MAX=${LOAD_MAX:-24} \
EXT_CPU_LIMIT=${EXT_CPU_LIMIT:-1000} ABORT_SAMPLES=${ABORT_SAMPLES:-24} MAX_RETRIES=${MAX_RETRIES:-200}"
run() { echo "[$(date '+%F %T')] === $1"; env $G ./refreeze_cell.sh "$1" "$LOG" "$2" "$CORE" -- "$3"; }

run impulse_linreg_flat 40m "$JULIA_ENV IMPULSE_MH_ARM=flat   julia -t 1 --project=. impulse_mh_linreg_harness.jl"
run impulse_linreg_region 40m "$JULIA_ENV IMPULSE_MH_ARM=region julia -t 1 --project=. impulse_mh_linreg_harness.jl"
run impulse_outlier  60m "$JULIA_ENV julia -t 1 --project=. impulse_outlier_mh_harness.jl"
run impulse_linesine 40m "$JULIA_ENV julia -t 1 --project=. impulse_linesine_mh_harness.jl"
run impulse_coal     40m "$JULIA_ENV julia -t 1 --project=. impulse_coal_mh_masked.jl"
run impulse_scaledcov_sicm_triple 90m \
  "$JULIA_ENV E2_OUT=$LOG/out/scaledcov julia -t 1 --project=$B $ROOT/test/probprog/mh_sicm_benchmark.jl"
EV200="${EV200:-/mnt/sbrantq/probprog/oopsla/evidence/refreeze_2026-09-08/fig13_generality/scaledcov_n200}"
mkdir -p "$EV200"
run impulse_scaledcov_sicm_triple_n200 120m \
  "$JULIA_ENV E2_N=200 E2_ITERS=500 E2_OUT=$EV200 julia -t 1 --project=$B $ROOT/test/probprog/mh_sicm_benchmark.jl"

# The Gen.jl and GenJAX cells are carried over unchanged from the pass this
# replaces; copy them in so the figure's ledger is one directory.
for c in gen_coal gen_linesine gen_linreg_and_scaledcov gen_outlier \
         genjax_coal genjax_linesine genjax_linreg genjax_outlier \
         genjax_scaledcov genjax_thread_probe; do
  cp -a "$B/logs/refreeze_2026-09-08/fig13_impulse_no_eigen_flag/$c.log" "$LOG/" 2>/dev/null
  cp -a "$B/logs/refreeze_2026-09-08/fig13_impulse_no_eigen_flag/cells/$c."* "$LOG/cells/" 2>/dev/null
done
cp -a "$B/logs/refreeze_2026-09-08/fig13_impulse_no_eigen_flag/out/." "$LOG/out/" 2>/dev/null
echo "[$(date '+%F %T')] symmetric Impulse generality pass done"
