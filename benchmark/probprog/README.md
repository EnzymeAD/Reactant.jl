# PPLBench: Impulse Benchmark Suite

Multi-framework NUTS benchmark infrastructure comparing **Impulse** (Reactant.jl)
against NumPyro, Turing.jl, and Stan on standard probabilistic models.

## Quick Start

```bash
# Run a full benchmark (all frameworks, 3 trials):
./run_pplbench.sh standard/logistic_regression

# Validate correctness (NumPyro vs Impulse, bitwise agreement):
./run_pplbench.sh --validate standard/logistic_regression

# Specific device:
./run_pplbench.sh --device cuda standard/logistic_regression
```

## Available Models

| Config | Model | Parameters | Data |
|--------|-------|------------|------|
| `standard/logistic_regression` | Bayesian Logistic Regression | alpha (1), beta (10) | n=2000, k=10 |
| `standard/gaussian_process` | GP Regression (SE kernel) | kernel_var, kernel_length, kernel_noise | n=60 |
| `standard/gaussian_process_small` | GP Regression (small run) | same as above | n=60, fewer iterations |
| `motivating/gp_pois_regr` | GP Poisson Regression (NCP) | alpha (1), f_tilde (11) | n=11, fixed rho=6.0 |

## Frameworks

| Framework | Language | Compilation | Harness | Init Params |
|-----------|----------|-------------|---------|-------------|
| **Impulse** | Julia (Reactant.jl) | AOT (XLA) | `harness.jl` | Constraint tensor |
| **NumPyro** | Python (JAX) | JIT (XLA) | in-process | Dict |
| **Turing.jl** | Julia | JIT (Julia) | `turing_harness.jl` | NamedTuple via `InitFromParams` |
| **Stan** | Stan → C++ | AOT (C++) | `stan_harness.py` | Dict |

### Timing Methodology

- **AOT frameworks (Impulse, Stan):** Compilation is a distinct phase. `compile_time` and
  `run_time` are cleanly separated.
- **Turing.jl:** A small warmup `sample()` call triggers Julia JIT before the first timed
  trial. JIT time is reported as `compile_time`. `_is_jit = True` additionally discards
  trial 0 as a safety net.
- **NumPyro:** JAX JIT is inseparable from the first inference call. `compile()` is a
  no-op. Trial 0 timing is discarded entirely (`_is_jit = True`).

## Architecture

```
benchmark/probprog/
├── run_pplbench.sh              # Entry point
├── harness.jl                   # Impulse subprocess server
├── turing_harness.jl            # Turing.jl subprocess server
├── stan_harness.py              # Stan subprocess server
├── numpyro_harness.py           # NumPyro reference (standalone)
│
├── standard/                    # Impulse model specs
│   ├── logistic_regression.jl
│   └── gaussian_process.jl
├── motivating/
│   └── gp_pois_regr.jl
│
├── turing/                      # Turing.jl model specs
│   ├── standard/
│   │   ├── logistic_regression.jl
│   │   └── gaussian_process.jl
│   └── motivating/
│       └── gp_pois_regr.jl
│
├── turing_env/                  # Turing.jl Julia environment
│   ├── Project.toml
│   └── Manifest.toml
│
├── pplbench_configs/            # Benchmark configurations
│   ├── standard/
│   │   ├── logistic_regression.json
│   │   ├── gaussian_process.json
│   │   └── gaussian_process_small.json
│   └── motivating/
│       └── gp_pois_regr.json
│
├── pplbench/                    # Python benchmark framework
│   └── pplbench/
│       ├── __main__.py          # CLI entry point
│       ├── lib/ppl_helper.py    # Core benchmark loop
│       ├── models/              # Data generators
│       │   ├── logistic_regression.py
│       │   ├── gaussian_process.py
│       │   └── gp_pois_regr.py
│       └── ppls/                # Framework adapters
│           ├── subprocess_inference.py  # Base class for subprocess PPLs
│           ├── impulse/         # Impulse adapter
│           ├── numpyro/         # NumPyro adapter
│           ├── turing/          # Turing.jl adapter
│           └── stan/            # Stan adapter
│
├── collect_paper_data.py        # Collect data for paper tables
├── generate_tables.py           # Generate LaTeX tables from paper_data.json
└── outputs/                     # Benchmark results (timestamped)
```

### Subprocess Protocol

Impulse, Turing.jl, and Stan use a persistent subprocess server for multi-trial benchmarks:

1. Python starts the subprocess with `--server` flag
2. Subprocess runs the first trial (includes compilation), writes output JSON
3. Subprocess prints `###READY###` to stdout
4. For each subsequent trial, Python sends `{"seed": N, "output": "path.json"}` via stdin
5. Subprocess runs the trial, writes output, prints `###DONE### <time_ms>`
6. Python sends `EXIT` to terminate

### Model Spec Contract

Each model file (e.g., `standard/logistic_regression.jl`) defines:

**Impulse models** (`harness.jl`):
- `setup(data)` → `(model_fn, model_args, selection, position_size, model_name)`
- `build_constraint(data, init_params)` → `ProbProg.Constraint`
- `extract_samples(trace)` → `Dict{String,Any}`

**Turing models** (`turing_harness.jl`):
- `setup(data)` → `(turing_model, model_name)`
- `get_init_params(data, init_params)` → NamedTuple or nothing
- `extract_samples(chain, num_samples)` → `Dict{String,Any}`

## Paper Data Pipeline

```bash
# 1. Run correctness validation only:
python collect_paper_data.py --validate-only

# 2. Run full benchmarks (all models, all frameworks):
python collect_paper_data.py

# 3. Or read from existing benchmark outputs:
python collect_paper_data.py --from-outputs outputs/2026-03-02_16:40:50

# 4. Generate LaTeX tables:
python generate_tables.py  # reads paper_data.json → paper/tables/*.tex
```

Output files:
- `paper_data.json` — aggregated correctness + timing data
- `paper/tables/correctness.tex` — numerical agreement table
- `paper/tables/baselines.tex` — baseline performance table

## Adding a New Model

1. **Data generator**: Create `pplbench/pplbench/models/my_model.py` (extends `BaseModel`)
2. **Impulse spec**: Create `standard/my_model.jl` (or `motivating/`)
3. **NumPyro spec**: Create `pplbench/pplbench/ppls/numpyro/my_model.py`
4. **Turing spec**: Create `turing/standard/my_model.jl`
5. **Python adapters**: Create per-framework adapter files (e.g., `ppls/impulse/my_model.py`)
6. **Config**: Create `pplbench_configs/standard/my_model.json`
7. **Table gen**: Update `generate_tables.py` display names and config mappings

## Adding a New Framework

1. Create a harness script (subprocess server following `###READY###/###DONE###/EXIT` protocol)
2. Create `pplbench/pplbench/ppls/<framework>/` with:
   - `__init__.py`
   - `base_<framework>_impl.py` (abstract base with `extract_data_from_<framework>`)
   - `inference.py` (extends `SubprocessMCMC`, sets `_compile_time_attr` and `_is_jit`)
   - Per-model adapter files
3. Add `<framework>_compile_time` to `ppl_helper.py` line 238
4. Add framework entries to config JSONs
5. Update `generate_tables.py` `FRAMEWORK_ORDER`

## Flags

| Flag | Description |
|------|-------------|
| `--validate` | Correctness mode: 5 samples, no warmup, no adaptation, compare NumPyro vs Impulse |
| `--device cpu\|cuda` | Execution device |
| `--ppls numpyro,impulse` | Run only the listed frameworks (comma-separated); skips optional entries |
| `--ppls-all numpyro,impulse` | Like `--ppls` but includes optional entries (e.g. "Impulse (no opt)") |
| `--dump-mlir` | Dump MLIR/StableHLO IR for Impulse |
| `--profile` | Enable XLA profiling (view with `python serve_traces.py`) |
| `--profile-breakdown` | Per-op runtime percentage breakdown |

### Optional PPL Entries

Config entries with `"optional": true` (e.g., "Impulse (no opt)") are **always skipped**
unless `--ppls-all` is used:

```bash
# Default: skips "Impulse (no opt)"
./run_pplbench.sh sicm/hierarchical_mvn

# --ppls also skips optional entries:
./run_pplbench.sh --ppls numpyro,impulse sicm/hierarchical_mvn

# --ppls-all includes optional entries:
./run_pplbench.sh --ppls-all numpyro,impulse sicm/hierarchical_mvn
```
