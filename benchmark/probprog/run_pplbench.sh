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
#   --threads N         Set intra-op thread count for all frameworks (default: 32)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE="cpu"
CONFIG_NAME=""
VALIDATE=""
PPLS_FILTER=""
NTHREADS=32

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

# Thread configuration: enforce uniform intra-op parallelism across all frameworks.
# Pass --threads 1 for a single-threaded pure algorithmic/compiler comparison.

# Julia (Turing.jl, Impulse)
export JULIA_NUM_THREADS=$NTHREADS

# BLAS backends (PyMC via OpenBLAS, Turing via Julia's OpenBLAS)
export OPENBLAS_NUM_THREADS=$NTHREADS
export MKL_NUM_THREADS=$NTHREADS
export OMP_NUM_THREADS=$NTHREADS

# XLA — Eigen threadpool.
# --xla_cpu_multi_thread_eigen is understood by both JAX's XLA and Reactant's XLA.
# --intra_op_parallelism_threads is JAX-only; Reactant's XLA fatally rejects it.
# We set the shared flag here; the JAX-only flag is set in numpyro_harness.py.
if [ "$NTHREADS" -eq 1 ]; then
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_cpu_multi_thread_eigen=false"
else
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_cpu_multi_thread_eigen=true"
fi

# Stan / CmdStan (TBB threadpool, reduce_sum parallelism)
export STAN_NUM_THREADS=$NTHREADS
export TBB_NUM_THREADS=$NTHREADS

# PyTensor BLAS: use conda openblas for PyMC performance.
# Only set PYTENSOR_FLAGS (not LD_LIBRARY_PATH, which would pollute Stan/clang).
# The pymc_harness.py sets LD_LIBRARY_PATH for its own subprocess.
CONDA_LIB="$(dirname "$PYTHON")/../lib"
export PPLBENCH_CONDA_LIB="${CONDA_LIB}"
PYTENSOR_EXTRA="mode=FAST_RUN"
if [ "$NTHREADS" -eq 1 ]; then
    PYTENSOR_EXTRA="${PYTENSOR_EXTRA},openmp=False"
else
    PYTENSOR_EXTRA="${PYTENSOR_EXTRA},openmp=True,openmp_elemwise_minsize=10"
fi
export PYTENSOR_FLAGS="blas__ldflags=-L${CONDA_LIB} -lopenblas,${PYTENSOR_EXTRA}"

export PYTHONPATH="${SCRIPT_DIR}/pplbench:${PYTHONPATH:-}"

EXTRA_ARGS=()
[ -n "$VALIDATE" ] && EXTRA_ARGS+=(--validate)
[ -n "$PPLS_FILTER" ] && EXTRA_ARGS+=("${PPLS_FLAG:-"--ppls"}" "$PPLS_FILTER")

if [ -n "$VALIDATE" ]; then
    echo "Running validation (NumPyro vs Impulse, 5 samples, no warmup, no adaptation)..."
else
    echo "Running benchmark..."
fi
"$PYTHON" -m pplbench "$CONFIG_FILE" "${EXTRA_ARGS[@]}"

if [ -n "${PPLBENCH_PROFILE:-}" ]; then
    echo ""
    echo "=== View traces in Perfetto ==="
    echo "  python $SCRIPT_DIR/serve_traces.py"
    echo "  (on remote: ssh -L 9001:localhost:9001 <host>)"
fi
