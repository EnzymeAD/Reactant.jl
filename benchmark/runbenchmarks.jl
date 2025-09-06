using BenchmarkTools: BenchmarkTools, BenchmarkGroup, @btime, @benchmarkable
using InteractiveUtils: versioninfo
using Reactant: Reactant
using Statistics: median

# To run benchmarks on a specific GPU backend, add AMDGPU / CUDA / Metal / oneAPI
# to benchmarks/Project.toml and change BENCHMARK_GROUP to the backend name
BENCHMARK_GROUP = get(ENV, "BENCHMARK_GROUP", nothing)

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5

if BENCHMARK_GROUP == "CUDA"
    Reactant.set_default_backend("gpu")
    @info "Running CUDA benchmarks" maxlog = 1
elseif BENCHMARK_GROUP == "TPU"
    Reactant.set_default_backend("tpu")
elseif BENCHMARK_GROUP == "CPU"
    Reactant.set_default_backend("cpu")
    @info "Running CPU benchmarks" maxlog = 1
else
    BENCHMARK_GROUP = String(split(string(first(Reactant.devices())), ":")[1])
    @info "Running $(BENCHMARK_GROUP) benchmarks" maxlog = 1
end

@assert BENCHMARK_GROUP in ("CPU", "CUDA", "TPU")

const SUITE = BenchmarkGroup()

# Main benchmark files
include("setup.jl")
setup_benchmarks!(SUITE, BENCHMARK_GROUP)

results = BenchmarkTools.run(SUITE; verbose=true)

filepath = joinpath(dirname(@__FILE__), "results")
mkpath(filepath)
filename = string(BENCHMARK_GROUP, "benchmarks.json")
BenchmarkTools.save(joinpath(filepath, filename), median(results))

@info "Saved results to $(joinpath(filepath, filename))"
