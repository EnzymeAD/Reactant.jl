using InteractiveUtils: versioninfo
using PrettyTables: pretty_table
using Reactant: Reactant
using JSON3: JSON3
using PrettyTables: pretty_table

@info sprint(io -> versioninfo(io; verbose=true))

# To run benchmarks on a specific backend
BENCHMARK_GROUP = get(ENV, "BENCHMARK_GROUP", nothing)

if BENCHMARK_GROUP == "CUDA"
    Reactant.set_default_backend("gpu")
    @info "Running CUDA benchmarks" maxlog = 1
elseif BENCHMARK_GROUP == "TPU"
    Reactant.set_default_backend("tpu")
elseif BENCHMARK_GROUP == "CPU"
    Reactant.set_default_backend("cpu")
    @info "Running CPU benchmarks" maxlog = 1
else
    BENCHMARK_GROUP = String(split(string(first(Reactant.devices())), ":")[1])
    @info "Running $(BENCHMARK_GROUP) benchmarks" maxlog = 1
end

@assert BENCHMARK_GROUP in ("CPU", "CUDA", "TPU")

# Main benchmark files
include("setup.jl")

results = run_benchmarks(BENCHMARK_GROUP)

table = Matrix{Any}(undef, length(results), 5)
for (i, (k, v)) in enumerate(sort(results))
    i1, i2, i3, i4 = rsplit(k, "/"; limit=4)
    table[i, 1] = i1
    table[i, 2] = i2
    table[i, 3] = i3
    table[i, 4] = i4
    table[i, 5] = v
end

pretty_table(
    table;
    alignment=[:l, :l, :l, :l, :c],
    column_labels=["Benchmark", "Mode", "Backend", "Passes", "Time (s)"],
    display_size=(-1, -1),
)

filepath = joinpath(dirname(@__FILE__), "results")
mkpath(filepath)
filename = string(BENCHMARK_GROUP, "benchmarks.json")

standardized_results = Vector{Dict{String,Union{String,Float64}}}(
    undef, length(keys(results))
)
for (i, (k, v)) in enumerate(results)
    standardized_results[i] = Dict("name" => k, "value" => v, "unit" => "s")
end

open(joinpath(filepath, filename), "w") do io
    return JSON3.pretty(io, JSON3.write(standardized_results))
end

@info "Saved results to $(joinpath(filepath, filename))"
