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

all_results = run_all_benchmarks(BENCHMARK_GROUP)

# Display results in a table
if !isempty(all_results["Runtime (s)"])
    table = Matrix{Any}(undef, length(all_results["Runtime (s)"]), 6)
    for (i, (k, v)) in enumerate(sort(all_results["Runtime (s)"]))
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
        table[i, 6] = all_results["TFLOP/s"][k]
    end

    pretty_table(
        table;
        alignment=[:l, :l, :l, :l, :c, :c],
        column_labels=["Benchmark", "Mode", "Backend", "Passes", "Time (s)", "TFLOP/s"],
        display_size=(-1, -1),
    )
end

include("utils.jl")

save_results(all_results, joinpath(dirname(@__FILE__), "results"), "", BENCHMARK_GROUP)
