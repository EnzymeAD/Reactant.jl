# Main Benchmark Runner
# This script runs benchmarks for a single subdirectory.
# Usage: julia --project=benchmark/<suite> benchmark/runbenchmarks.jl <backend> <suite>

using InteractiveUtils: versioninfo

@info sprint(io -> versioninfo(io; verbose=true))

# Parse arguments
@assert length(ARGS) == 2 "Usage: julia --project=benchmark/<suite> \
                           benchmark/runbenchmarks.jl <backend> <suite>"
const BENCHMARK_GROUP = ARGS[1]
const BENCHMARK_SUITE = ARGS[2]
@assert BENCHMARK_GROUP in ("CPU", "CUDA", "TPU") "Unknown backend: $(BENCHMARK_GROUP)"

suite_dir = joinpath(dirname(@__FILE__), BENCHMARK_SUITE)
@assert isdir(suite_dir) "Benchmark suite directory not found: $(suite_dir)"
@assert isfile(joinpath(suite_dir, "runbenchmarks.jl")) "No runbenchmarks.jl found in $(suite_dir)"

@info "Running $(BENCHMARK_GROUP) benchmarks for suite: $(BENCHMARK_SUITE)"

# Set BENCHMARK_GROUP env var so subdirectory scripts can pick it up via get_backend()
ENV["BENCHMARK_GROUP"] = BENCHMARK_GROUP

# Run the suite's benchmarks (includes pretty-printing and saving results)
include(joinpath(suite_dir, "runbenchmarks.jl"))
