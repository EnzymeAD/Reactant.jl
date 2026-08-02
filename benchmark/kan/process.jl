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

mem(x) = prettymemory(x.memory_data["GPU_1_bfc"].peak_bytes_usage_lifetime)
flops(x) = x.metrics_data.raw_flops / 1e9

df = @chain df begin
    @mutate mlp = bench_mlp_comp
    @mutate kan1 = bench_kan1_comp
    @mutate kan2 = bench_kan2_comp
    @mutate grad_mlp = bench_grad_mlp_comp
    @mutate grad_kan1 = bench_grad_kan1_comp
    @mutate grad_kan2 = bench_grad_kan2_comp
    @select N optimize mlp grad_mlp kan1 grad_kan1 kan2 grad_kan2
end
