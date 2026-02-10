# Main Benchmark Runner
# This script orchestrates running all benchmarks across subdirectories and aggregates results.

using InteractiveUtils: versioninfo
using PrettyTables: pretty_table
using JSON: JSON

@info sprint(io -> versioninfo(io; verbose=true))

# Determine backend from environment
@assert length(ARGS) == 1 "Usage: julia --project=benchmark benchmark/runbenchmarks.jl \
                           <backend>"
const BENCHMARK_GROUP = ARGS[1]
@assert BENCHMARK_GROUP in ("CPU", "CUDA", "TPU") "Unknown backend: $(BENCHMARK_GROUP)"

@info "Running $(BENCHMARK_GROUP) benchmarks"

# Main benchmark orchestration
include("setup.jl")

results = run_all_benchmarks(BENCHMARK_GROUP)

# Display results in a table
if !isempty(results)
    table = Matrix{Any}(undef, length(results), 5)
    for (i, (k, v)) in enumerate(sort(results))
        parts = rsplit(k, "/"; limit=4)
        # Fill in parts, padding with empty strings if fewer than 4 parts
        while length(parts) < 4
            push!(parts, "")
        end
        i1, i2, i3, i4 = parts[1], parts[2], parts[3], parts[4]
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
end

# Save aggregated results
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
    return JSON.json(io, standardized_results; pretty=true)
end

@info "Saved $(length(results)) results to $(joinpath(filepath, filename))"

# Also aggregate any results saved by individual subdirectories
@info "Aggregating results from subdirectories..."
subdirectory_results = aggregate_saved_results(dirname(@__FILE__))
if !isempty(subdirectory_results)
    combined_filename = string(BENCHMARK_GROUP, "_combined_benchmarks.json")
    open(joinpath(filepath, combined_filename), "w") do io
        return JSON.json(io, subdirectory_results; pretty=true)
    end
    @info "Saved $(length(subdirectory_results)) combined results to $(joinpath(filepath, combined_filename))"
end
