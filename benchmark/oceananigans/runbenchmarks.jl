# Oceananigans Benchmarks Runner
# This script runs all oceananigans benchmarks and stores results to a JSON file
using Reactant, LinearAlgebra, Enzyme
using Random: Random
using Printf: @sprintf

include("../utils.jl")

@info sprint(io -> versioninfo(io; verbose=true))

backend = get_backend()

# Include benchmark modules
module AbernatheyChannel

include("abernathey_channel.jl")

end

# Run all benchmarks
function run_all_benchmarks(backend::String)
    results = Dict{String,Dict{String,Float64}}()

    AbernatheyChannel.run_abernathey_channel_benchmark!(results, backend)

    return results
end

results = run_all_benchmarks(backend)

save_results(results, joinpath(@__DIR__, "results"), "oceananigans", backend)
