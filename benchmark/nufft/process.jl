using DataFrames
using TidierData
using FileIO
using JLD2
using BenchmarkTools
using BenchmarkTools: prettytime, prettymemory, time
using Reactant
using Statistics

df = DataFrame()

for Mexp in [5, 6, 7],
    N in [128, 256, 512, 1024, 2048, 4096],
    optimize in ["all", "after_enzyme"]

    jldopen("bench-1e$Mexp-$N-$optimize.jld") do data
        row = (; (Symbol(key) => data[key] for key in keys(data))...)
        push!(df, row)
    end
end

mem(x) = prettymemory(x.memory_data["GPU_1_bfc"].peak_bytes_usage_lifetime)
flops(x) = x.metrics_data.raw_flops / 1e9

df_primal = @chain df begin
    @filter M == 10_000_000 && N >= 256
    @mutate gflops = flops.(prof_primal_type_2)
    @mutate runtime = prettytime(time(mean.(bench_primal_type_2)))
    @mutate memory = mem.(prof_primal_type_2)
    @select N optimize gflops runtime memory
    @group_by optimize
end

df_reverse = @chain df begin
    @filter M == 10_000_000 && N >= 256
    @mutate gflops = flops.(prof_revdiff_enzyme)
    @mutate runtime = prettytime(time(mean.(bench_revdiff_enzyme)))
    @mutate memory = mem.(prof_revdiff_enzyme)
    @select N optimize gflops runtime memory
    @group_by optimize
end
