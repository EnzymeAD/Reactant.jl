using InteractiveUtils: versioninfo
using Reactant: Reactant
using JSON3: JSON3

using PProf, Profile

@info sprint(io -> versioninfo(io; verbose=true))

# To run benchmarks on a specific backend
BENCHMARK_GROUP = get(ENV, "BENCHMARK_GROUP", nothing)

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

# Main benchmark files
include("setup.jl")

Profile.clear()
results = @profile run_benchmarks(BENCHMARK_GROUP)
pprof()

filepath = joinpath(dirname(@__FILE__), "results")
mkpath(filepath)
filename = string(BENCHMARK_GROUP, "benchmarks.json")

standardized_results = Vector{Dict{String,Union{String,Float64}}}(
    undef, length(keys(results))
)
for (i, (k, v)) in enumerate(results)
    standardized_results[i] = Dict("name" => k, "time" => v, "unit" => "s")
end

open(joinpath(filepath, filename), "w") do io
    JSON3.pretty(io, JSON3.write(standardized_results))
end

@info "Saved results to $(joinpath(filepath, filename))"
