#!/bin/bash
#
# Generality figure (user-composed MH throughput) — single-core pinned pass.
#
# Every cell of every system runs on ONE physical core (taskset), with BLAS/OMP
# capped at one thread (uncapped OpenBLAS under a one-core pin spins ~192
# workers on that core and inflates Gen.jl cells ~1000x), sequentially, so the
# three systems see identical conditions. Statistic: MEDIAN over the in-process
# repetitions after one discarded warm-up repetition; the harnesses print every
# repetition's wall time ("... REP wall ... | per-move ... us") and the median is
# taken from those lines by summarize_generality.py.
#
# Usage: run_generality_pinned.sh [core] [logdir]
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CORE="${1:-48}"
LOGDIR="${2:-$SCRIPT_DIR/logs/refreeze_2026-09-08/fig13}"
mkdir -p "$LOGDIR/cells" "$LOGDIR/out"
CELL="$SCRIPT_DIR/refreeze_cell.sh"

# The single-core protocol is symmetric across systems: run_pplbench.sh sets
# XLA_FLAGS=--xla_cpu_multi_thread_eigen=false for EVERY framework at
# --threads 1, so the Impulse cells get it here too. Until 2026-09-08 this
# driver gave the flag only to the GenJAX cells, which left the two systems on
# different thread settings; the Impulse cells measured without it are kept in
# the ledger under `impulse_no_eigen_flag` for comparison. Gen.jl has no XLA and
# is unaffected either way.
JULIA_ENV='CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false'
GENJAX_PY=/home/sbrantq/anaconda3/envs/genjax-bench/bin/python
JAX_ENV='CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false'

run() {  # run <label> <timeout> <command string>
  echo "[$(date '+%F %T')] === cell $1"
  "$CELL" "$1" "$LOGDIR" "$2" "$CORE" -- "$3"
}

# ---- thread-confirmation probe (records affinity + OS thread count for jax) --
cat > "$LOGDIR/jax_thread_probe.py" <<'PY'
import os, jax, jax.numpy as jnp
def nthreads():
    for l in open("/proc/self/status"):
        if l.startswith("Threads:"):
            return int(l.split()[1])
x = jnp.ones((512, 512)); (x @ x).block_until_ready()
print("affinity:", sorted(os.sched_getaffinity(0)),
      "| jax devices:", jax.devices(),
      "| local_device_count:", jax.local_device_count(),
      "| OS threads:", nthreads())
PY
run genjax_thread_probe 10m "$JAX_ENV $GENJAX_PY $LOGDIR/jax_thread_probe.py"

# ---- Impulse -----------------------------------------------------------------
run impulse_linreg_flat   40m "$JULIA_ENV IMPULSE_MH_ARM=flat   julia -t 1 --project=. impulse_mh_linreg_harness.jl"
run impulse_linreg_region 40m "$JULIA_ENV IMPULSE_MH_ARM=region julia -t 1 --project=. impulse_mh_linreg_harness.jl"
run impulse_outlier       60m "$JULIA_ENV julia -t 1 --project=. impulse_outlier_mh_harness.jl"
run impulse_linesine      40m "$JULIA_ENV julia -t 1 --project=. impulse_linesine_mh_harness.jl"
run impulse_coal          40m "$JULIA_ENV julia -t 1 --project=. impulse_coal_mh_masked.jl"

# scaledcov + the isolated-SICM triple (SICM region / region-no-SICM / no-opt flat)
# at n=8, the size of the figure's scaled-covariance row.
run impulse_scaledcov_sicm_triple 90m \
  "$JULIA_ENV E2_OUT=$LOGDIR/out/scaledcov julia -t 1 --project=$SCRIPT_DIR $ROOT/test/probprog/mh_sicm_benchmark.jl"

# The same triple at n=200, the size the SICM ablation figure uses. At n=8 the
# factorization is a cheap custom_call, so the ratio there says nothing about
# what the rewrite is worth; the paper's claim is that SICM applies inside
# user-composed kernels, and this is the size that tests it. The figure's row
# stays at n=8; this cell is reported separately.
EV_N200="/mnt/sbrantq/probprog/oopsla/evidence/refreeze_2026-09-08/fig13_generality/scaledcov_n200"
mkdir -p "$EV_N200"
run impulse_scaledcov_sicm_triple_n200 120m \
  "$JULIA_ENV E2_N=200 E2_ITERS=500 E2_OUT=$EV_N200 julia -t 1 --project=$SCRIPT_DIR $ROOT/test/probprog/mh_sicm_benchmark.jl"

# ---- Gen.jl ------------------------------------------------------------------
run gen_linreg_and_scaledcov 40m "$JULIA_ENV GEN_MH_OUT=$LOGDIR/out/gen julia -t 1 --project=gen_env gen_mh_harness.jl"
run gen_outlier              30m "$JULIA_ENV julia -t 1 --project=gen_env gen_outlier_mh_harness.jl"
run gen_linesine             30m "$JULIA_ENV julia -t 1 --project=gen_env gen_linesine_mh_harness.jl"
run gen_coal                 30m "$JULIA_ENV julia -t 1 --project=gen_env gen_coal_mh_vec_harness.jl"

# ---- GenJAX ------------------------------------------------------------------
run genjax_linreg    30m "$JAX_ENV $GENJAX_PY genjax_mh_linreg_harness.py"
run genjax_outlier   40m "$JAX_ENV $GENJAX_PY genjax_outlier_mh_harness.py"
run genjax_linesine  30m "$JAX_ENV $GENJAX_PY genjax_linesine_mh_harness.py"
run genjax_coal      30m "$JAX_ENV $GENJAX_PY genjax_coal_mh_harness.py"
run genjax_scaledcov 40m "$JAX_ENV GENJAX_MH_OUT=$LOGDIR/out $GENJAX_PY genjax_mh_harness.py"

echo "[$(date '+%F %T')] generality pass done; logs in $LOGDIR"
