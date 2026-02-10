# Polybench Benchmarks Runner
# This script runs all polybench benchmarks and stores results to a JSON file

include("common.jl")

@info sprint(io -> versioninfo(io; verbose=true))

backend = get_backend()

# Load dependencies used in benchmarks
using Reactant, LinearAlgebra
using Chairmarks: @b
using Random: Random
using Printf: @sprintf

# Include benchmark modules
include("stencil.jl")
include("data_mining.jl")
include("blas.jl")
include("linalg_kernels.jl")

# Run all benchmarks
function run_all_benchmarks(backend::String)
    results = Dict{String,Dict{String,Float64}}()

    run_data_mining_benchmarks!(results, backend)
    run_blas_benchmarks!(results, backend)
    run_linalg_kernel_benchmarks!(results, backend)
    run_stencil_benchmarks!(results, backend)

    return results
end

results = run_all_benchmarks(backend)

save_results(results, joinpath(@__DIR__, "results"), "polybench", backend)
