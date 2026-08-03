using DataFrames
using TidierData
using FileIO
using JLD2
using BenchmarkTools
using BenchmarkTools: prettytime, prettymemory, time
using Reactant
using Statistics
using Latexify
using LaTeXStrings
using Printf
using Plots
using StatsPlots
using CategoricalArrays

df = DataFrame()

for file in filter(endswith(".jld"), readdir(@__DIR__))
    jldopen(joinpath(@__DIR__, file)) do data
        row = (; (Symbol(key) => data[key] for key in keys(data))...)
        push!(df, row)
    end
end

mem(x) = prettymemory(only(values(x.memory_data)).peak_stats.peak_bytes_in_use)
flops(x) = x.metrics_data.raw_flops / 1e9
# prettybench(x) = @sprintf "%.1f" (time(mean(x)) / 1e3) # * " ± " * prettytime(time(std(x)))
prettybench(x) = time(mean(x)) / 1e3

function mapprop_sym(x)
    if x === :up
        "↑" # "\\uparrow"
    elseif x === :down
        "↓" # "\\downarrow"
    else
        "-"
    end
end

function mapprop(x)
    if x isa Symbol
        y = mapprop_sym(x)
        "($y, $y)"
    else
        "($(mapprop_sym(x.pre_ad)), $(mapprop_sym(x.post_ad)))"
    end# |> LaTeXString
end

function mapopt(x)
    if x === :after_enzyme
        "post"
    elseif x === :before_enzyme
        "pre"
    else
        "both"
    end
end

data = @chain df begin
    @mutate mlp = prettybench(bench_mlp_comp)
    @mutate kan1 = prettybench(bench_kan1_comp)
    @mutate kan2 = prettybench(bench_kan2_comp)
    @mutate grad_mlp = prettybench(bench_grad_mlp_comp)
    @mutate grad_kan1 = prettybench(bench_grad_kan1_comp)
    @mutate grad_kan2 = prettybench(bench_grad_kan2_comp)
    @mutate optimize = mapopt(getproperty(compile_options, :optimization_passes))
    @mutate reshape_propagate = mapprop(getproperty(compile_options, :reshape_propagate))
    @select optimize reshape_propagate grad_kan1 grad_kan2
end

# latexify(data; env = :table, booktabs = true, latex = true)

function map_sorterprop(x)
    if x === "(-, -)"
        1
    elseif x === "(↑, ↑)"
        2
    else
        3
    end
end

sort!(data, [order(:optimize, rev=true), order(:reshape_propagate, by=map_sorterprop)])

groupedbar(
    data.optimize,
    data.grad_kan1,
    group = data.reshape_propagate,
    ylim=(80,300),
    ylabel = LaTeXString("Runtime [\\mus]"),
    title = "KAN 1",
)

groupedbar(
    data.optimize,
    data.grad_kan2,
    group = data.reshape_propagate,
    # ylim=(80,300),
    ylabel = LaTeXString("Runtime [\\mus]"),
    title = "KAN 2",
)
