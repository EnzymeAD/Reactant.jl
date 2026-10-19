# Comrade Benchmarks Runner
# This script runs all Comrade benchmarks and stores results to a JSON file
# Load dependencies
using Reactant
using Chairmarks: @b
using Random: Random
using Printf: @sprintf

using LinearAlgebra
using Accessors: @set, @reset
using VLBISkyModels
using VLBILikelihoods
using Comrade
using Distributions
using VLBIImagePriors
using LogExpFunctions
import TransformVariables as TV
using VLBIFiles

using Downloads
using Distributions
using Enzyme

using Test

include("common.jl")

@info sprint(io -> versioninfo(io; verbose=true))

backend = get_backend()

include("comimager.jl")

function run_all_benchmarks(backend::String)
    results = Dict{String,Dict{String,Float64}}()

    dataurl = "https://github.com/ptiede/ComradeTestData/releases/download/Data/eht_2017_data.uvfits"
    dataf = Base.download(dataurl)

    T = backend == "TPU" ? Float32 : Float64

    for sz in (64, 128, 256)
        tpostr = build_post(T, μas2rad(200.0), sz, dataf)
        run_comrade_benchmark!(
            results, "Comrade EHT Imaging $(sz) x $(sz) [$(T)]", backend, tpostr
        )
    end

    return results
end

results = run_all_benchmarks(backend)

save_results(results, joinpath(@__DIR__, "results"), "comrade", backend)
pretty_print_results(results, "comrade", backend)
