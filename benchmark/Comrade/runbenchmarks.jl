# Comrade Benchmarks Runner
# This script runs all Comrade benchmarks and stores results to a JSON file

using Pkg

@static if VERSION ≥ v"1.10-" && VERSION < v"1.11"
    Pkg.add([
        PackageSpec(; name="ComradeBase", rev="main"),
        PackageSpec(; name="Comrade", rev="ptiede-reactant"),
        PackageSpec(; name="VLBISkyModels", rev="ptiede-reactnfft"),
        PackageSpec(; name="VLBILikelihoods", rev="ptiede-reactant"),
        PackageSpec(; name="VLBIImagePriors", rev="ptiede-reactantperf"),
        PackageSpec(;
            url="https://github.com/ptiede/TransformVariables.jl", rev="ptiede-reactant"
        ),
        PackageSpec(;
            url="https://github.com/ptiede/NFFT.jl", rev="ptiede-reactant"
        )
    ])
    Pkg.develop(; path=joinpath(@__DIR__, "../../"))
end


# Load dependencies
using Reactant
using Chairmarks: @b
using Random: Random
using Printf: @sprintf



using Reactant
using LinearAlgebra
using AbstractFFTs

using VLBISkyModels
using VLBILikelihoods
using Comrade
using Distributions
using VLBIImagePriors
using LogExpFunctions
import TransformVariables as TV

using Downloads
using Distributions
using Enzyme

using Pyehtim
using Test

include("common.jl")

@info sprint(io -> versioninfo(io; verbose=true))

include("comimager.jl")

function run_all_benchmarks(backend::String)
    results = Dict{String,Dict{String,Float64}}()

    tpostr = build_post(μas2rad(200.0), 64, dataf)
    run_comrade_benchmark!(results, "Comrade EHT Imaging 64 x 64", backend, tpostr)

    return results
end


res = run_all_benchmarks(get_backend())