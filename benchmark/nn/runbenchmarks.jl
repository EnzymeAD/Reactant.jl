# Neural Network Benchmarks Runner
# This script runs all NN benchmarks and stores results to a JSON file

include("common.jl")

@info sprint(io -> versioninfo(io; verbose=true))

backend = get_backend()

# Include benchmark modules
include("vision.jl")
include("neural_operators.jl")
include("dgcnn.jl")

# Run all benchmarks
function run_all_benchmarks(backend::String)
    results = Dict{String,Dict{String,Float64}}()

    # Vision models
    run_vgg_benchmark!(results, backend)
    run_vit_benchmark!(results, backend)

    # Neural operators
    run_deeponet_benchmark!(results, backend)
    run_fno_benchmark!(results, backend)

    # Point cloud models
    run_dgcnn_benchmark!(results, backend)

    return results
end

results = run_all_benchmarks(backend)

save_results(results, joinpath(@__DIR__, "results"), "nn", backend)
pretty_print_results(results, "nn", backend)
