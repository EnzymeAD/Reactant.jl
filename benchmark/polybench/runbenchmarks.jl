# Polybench Benchmarks Runner
# This script runs all polybench benchmarks and stores results to a JSON file
using InteractiveUtils: versioninfo
using Reactant: Reactant
using JSON3: JSON3
using Printf: @sprintf

@info sprint(io -> versioninfo(io; verbose=true))

# Get backend from environment or auto-detect
function get_backend()
    BENCHMARK_GROUP = get(ENV, "BENCHMARK_GROUP", nothing)

    if BENCHMARK_GROUP == "CUDA"
        Reactant.set_default_backend("gpu")
        @info "Running CUDA benchmarks" maxlog = 1
    elseif BENCHMARK_GROUP == "TPU"
        Reactant.set_default_backend("tpu")
        @info "Running TPU benchmarks" maxlog = 1
    elseif BENCHMARK_GROUP == "CPU"
        Reactant.set_default_backend("cpu")
        @info "Running CPU benchmarks" maxlog = 1
    else
        BENCHMARK_GROUP = String(split(string(first(Reactant.devices())), ":")[1])
        @info "Running $(BENCHMARK_GROUP) benchmarks" maxlog = 1
    end

    @assert BENCHMARK_GROUP in ("CPU", "CUDA", "TPU")
    return BENCHMARK_GROUP
end

backend = get_backend()

# Load dependencies used in benchmarks
using Reactant, LinearAlgebra
using Chairmarks: @b
using Random: Random

# Include benchmark modules
include("common.jl")
include("stencil.jl")
include("data_mining.jl")
include("blas.jl")
include("linalg_kernels.jl")

# Run all benchmarks
function run_all_benchmarks(backend::String)
    results = Dict{String,Float64}()

    run_data_mining_benchmarks!(results, backend)
    run_blas_benchmarks!(results, backend)
    run_linalg_kernel_benchmarks!(results, backend)
    run_stencil_benchmarks!(results, backend)

    return results
end

results = run_all_benchmarks(backend)

# Save results
results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)

filename = string("polybench_", backend, "_benchmarks.json")
filepath = joinpath(results_dir, filename)

standardized_results = Vector{Dict{String,Union{String,Float64}}}(undef, length(results))
for (i, (k, v)) in enumerate(results)
    standardized_results[i] = Dict("name" => k, "value" => v, "unit" => "s")
end

open(filepath, "w") do io
    JSON3.pretty(io, JSON3.write(standardized_results))
end

@info "Saved $(length(results)) benchmark results to $(filepath)"
