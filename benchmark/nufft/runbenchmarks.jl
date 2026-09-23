# NUFFT Benchmarks Runner
# This script runs all NUFFT benchmarks and stores results to a JSON file

include("common.jl")

@info sprint(io -> versioninfo(io; verbose=true))

backend = get_backend()

function run_all_benchmarks(backend::String)
    results = Dict{String,Dict{String,Float64}}()

    T = backend == "TPU" ? Float32 : Float64

    # One modest problem size: the analytic type-1 adjoint runs for minutes per
    # execution on the CPU backend, so bigger sizes belong in `sweep.jl`.
    run_nufft_benchmark!(results, backend; M=100_000, N=128, D=2, T)

    return results
end

results = run_all_benchmarks(backend)

save_results(results, joinpath(@__DIR__, "results"), "nufft", backend)
pretty_print_results(results, "nufft", backend)
