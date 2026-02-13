# Miscellaneous Benchmarks Runner
# This script runs all misc benchmarks and stores results to a JSON file

include("common.jl")

@info sprint(io -> versioninfo(io; verbose=true))

backend = get_backend()

# Load dependencies used in benchmarks
using Reactant, LinearAlgebra, Enzyme
using Chairmarks: @b
using Random: Random
using Printf: @sprintf

# Include benchmark modules
include("newton_schulz.jl")
include("bloch_rf_optimization.jl")

# Run all benchmarks
function run_all_benchmarks(backend::String)
    results = Dict{String,Dict{String,Float64}}()

    run_newton_schulz_benchmark!(results, backend)
    run_bloch_rf_optimization_benchmark!(results, backend)

    return results
end

results = run_all_benchmarks(backend)

save_results(results, joinpath(@__DIR__, "results"), "misc", backend)
pretty_print_results(results, "misc", backend)
