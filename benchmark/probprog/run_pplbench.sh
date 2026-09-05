#!/bin/bash
#
# Run PPLBench comparison between Impulse and other PPL frameworks.
#
# Usage:
#   ./run_pplbench.sh [flags] [category/config_name]
#
# Examples:
#   ./run_pplbench.sh standard/logistic_regression
#   ./run_pplbench.sh --validate standard/logistic_regression
#   ./run_pplbench.sh motivating/gp_pois_regr
#
# Flags:
#   --device cpu|cuda   Set execution device (default: cpu)
#   --ppls numpyro,impulse  Run only the listed frameworks (comma-separated)
#   --ppls-all numpyro,impulse  Like --ppls but includes optional entries (e.g. "no opt")
#   --dump-mlir         Dump MLIR/HLO IR
#   --profile           Enable XLA profiling
#   --profile-breakdown Per-op runtime percentage breakdown
#   --validate          Run NumPyro + Impulse with num_samples=5, num_warmup=0,
#                       no adaptation, compare outputs for numerical agreement
#   --threads N|all     Thread policy (default: all). "all" = fully uncapped, for
#                       Impulse/NumPyro-only runs (ablations). "1" = the
#                       cross-system comparison protocol: single-threaded,
#                       enforced by CPU affinity (use for any run that includes
#                       Stan/Turing). Numeric N>1 = affinity-enforced budget.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE="cpu"
CONFIG_NAME=""
VALIDATE=""
PPLS_FILTER=""
NTHREADS=all

# Use conda probprog-bench environment Python (has JAX, NumPyro, PyMC, etc.)
PYTHON="${PYTHON:-/home/sbrantq/anaconda3/envs/probprog-bench/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "Error: Python not found at $PYTHON"
    echo "Set PYTHON env var to your conda probprog-bench environment python."
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --dump-mlir)
            export PPLBENCH_DUMP_MLIR=1
            shift
            ;;
        --profile)
            export PPLBENCH_PROFILE=1
            shift
            ;;
        --profile-breakdown)
            export PPLBENCH_PROFILE_BREAKDOWN=1
            shift
            ;;
        --validate)
            VALIDATE=1
            shift
            ;;
        --ppls)
            PPLS_FLAG="--ppls"
            PPLS_FILTER="$2"
            shift 2
            ;;
        --ppls-all)
            PPLS_FLAG="--ppls-all"
            PPLS_FILTER="$2"
            shift 2
            ;;
        --threads)
            NTHREADS="$2"
            shift 2
            ;;
        *)
            CONFIG_NAME="$1"
            shift
            ;;
    esac
done

CONFIG_NAME="${CONFIG_NAME:-standard/logistic_regression}"
CONFIG_FILE="$SCRIPT_DIR/pplbench_configs/${CONFIG_NAME}.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    find "$SCRIPT_DIR/pplbench_configs" -name "*.json" -printf "  %P\n" 2>/dev/null | sed 's/\.json$//'
    exit 1
fi

echo "=== PPLBench: $CONFIG_NAME ==="
echo "Config: $CONFIG_FILE"
echo "Device: $DEVICE, Threads: $NTHREADS"
[ -n "${PPLBENCH_DUMP_MLIR:-}" ] && echo "MLIR dump: enabled"
[ -n "${PPLBENCH_PROFILE:-}" ] && echo "XLA profile: enabled"
[ -n "${PPLBENCH_PROFILE_BREAKDOWN:-}" ] && echo "Profile breakdown: enabled"
[ -n "$VALIDATE" ] && echo "Validate mode: enabled"
[ -n "$PPLS_FILTER" ] && echo "PPLs filter: $PPLS_FILTER"
echo ""

if [ "$DEVICE" = "cpu" ]; then
    export JAX_PLATFORM_NAME=cpu
    export CUDA_VISIBLE_DEVICES=""
elif [ "$DEVICE" = "cuda" ]; then
    export JAX_PLATFORM_NAME=gpu
else
    echo "Error: Unknown device '$DEVICE'. Use 'cpu' or 'cuda'."
    exit 1
fi

# Thread policy (2026-08-24, user decision):
#   all (default): no caps anywhere. XLA's Eigen pool sizes itself from the
#                  host, BLAS pools use library defaults, no CPU affinity.
#                  For Impulse/NumPyro-only experiments (ablations).
#   1:             cross-system protocol. Single-threaded, ENFORCED by taskset
#                  below (env vars alone cannot cap XLA: both JAX's and
#                  Reactant's XLA fatally reject --intra_op_parallelism_threads,
#                  audited 2026-08-23; the pool is sized from sched_getaffinity
#                  at client creation).
if [ "$NTHREADS" = "all" ]; then
    # Uncapped mode: inference.py falls back to "-t 32" when JULIA_NUM_THREADS
    # is unset, which caps only the Julia frontend (the measured sampling runs
    # inside the XLA executable, whose pool is sized from CPU affinity), but
    # "no cap" should hold to the letter: let julia size itself too.
    export JULIA_NUM_THREADS=auto
else
    # Julia (Turing.jl, Impulse)
    export JULIA_NUM_THREADS=$NTHREADS

    # BLAS backends (Turing via Julia's OpenBLAS)
    export OPENBLAS_NUM_THREADS=$NTHREADS
    export MKL_NUM_THREADS=$NTHREADS
    export OMP_NUM_THREADS=$NTHREADS

    # XLA multithreaded codegen off only at the single-thread budget.
    if [ "$NTHREADS" -eq 1 ]; then
        export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_cpu_multi_thread_eigen=false"
    else
        export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_cpu_multi_thread_eigen=true"
    fi

    # Stan / CmdStan (TBB threadpool, reduce_sum parallelism)
    export STAN_NUM_THREADS=$NTHREADS
    export TBB_NUM_THREADS=$NTHREADS
fi

# PyTensor BLAS: use conda openblas for PyMC performance.
# Only set PYTENSOR_FLAGS (not LD_LIBRARY_PATH, which would pollute Stan/clang).
# The pymc_harness.py sets LD_LIBRARY_PATH for its own subprocess.
CONDA_LIB="$(dirname "$PYTHON")/../lib"
export PPLBENCH_CONDA_LIB="${CONDA_LIB}"
PYTENSOR_EXTRA="mode=FAST_RUN"
if [ "$NTHREADS" = "1" ]; then
    PYTENSOR_EXTRA="${PYTENSOR_EXTRA},openmp=False"
else
    PYTENSOR_EXTRA="${PYTENSOR_EXTRA},openmp=True,openmp_elemwise_minsize=10"
fi
export PYTENSOR_FLAGS="blas__ldflags=-L${CONDA_LIB} -lopenblas,${PYTENSOR_EXTRA}"

export PYTHONPATH="${SCRIPT_DIR}/pplbench:${PYTHONPATH:-}"

EXTRA_ARGS=()
[ -n "$VALIDATE" ] && EXTRA_ARGS+=(--validate)
[ -n "$PPLS_FILTER" ] && EXTRA_ARGS+=("${PPLS_FLAG:-"--ppls"}" "$PPLS_FILTER")

# Enforce the thread budget with CPU affinity. XLA sizes its Eigen pool from
# sched_getaffinity, and the affinity is inherited by every framework subprocess
# (julia, CmdStan binaries, BLAS/OpenMP pools), so this caps all frameworks
# uniformly; the env vars above only shape how each pool uses the budget.
# CPUs 0-95 are distinct physical cores on this machine (SMT siblings are
# 96-191), so -c 0-(N-1) pins to N physical cores.
TASKSET=()
if [ "$NTHREADS" != "all" ]; then
    # getconf, NOT nproc: GNU nproc honors OMP_NUM_THREADS, which this script
    # exports above, so nproc would report the budget instead of the machine.
    NPROC=$(getconf _NPROCESSORS_ONLN)
    if [ "$NTHREADS" -lt "$NPROC" ]; then
        # PPLBENCH_TASKSET_CPUS lets concurrent single-threaded sessions use
        # disjoint cores (e.g. another agent pinned to CPU 60).
        CPUS="${PPLBENCH_TASKSET_CPUS:-0-$((NTHREADS-1))}"
        TASKSET=(taskset -c "$CPUS")
        echo "CPU affinity: ${TASKSET[*]}"
    fi
fi

if [ -n "$VALIDATE" ]; then
    echo "Running validation (NumPyro vs Impulse, 5 samples, no warmup, no adaptation)..."
else
    echo "Running benchmark..."
fi
"${TASKSET[@]}" "$PYTHON" -m pplbench "$CONFIG_FILE" "${EXTRA_ARGS[@]}"

if [ -n "${PPLBENCH_PROFILE:-}" ]; then
    echo ""
    echo "=== View traces in Perfetto ==="
    echo "  python $SCRIPT_DIR/serve_traces.py"
    echo "  (on remote: ssh -L 9001:localhost:9001 <host>)"
fi
