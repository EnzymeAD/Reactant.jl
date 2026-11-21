using DataFrames
using TidierData
using FileIO
using JLD2
using BenchmarkTools
using BenchmarkTools: prettytime, prettymemory, time
using Reactant
using Statistics

df = DataFrame()

for file in filter(endswith(".jld"), readdir(@__DIR__))
    jldopen(joinpath(@__DIR__, file)) do data
        row = (; (Symbol(key) => data[key] for key in keys(data))...)
        push!(df, row)
    end
end

sort!(df, :m)

mem(x) = prettymemory(only(values(x.profiling_result.memory_data)).peak_stats.peak_bytes_in_use)
flops(x) = x.profiling_result.metrics_data.raw_flops / 1e9

@chain df begin
    # @mutate gflops = flops.(@show(profile))
    @mutate runtime = prettytime(time(mean.(benchmark)))
    @mutate runtime_std = prettytime(time(std.(benchmark)))
    # @mutate memory = mem.(profile)
    @select m optimize runtime runtime_std
    # @group_by optimize
end
