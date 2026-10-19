# Common Benchmark Utilities
# Shared backend detection and result saving logic for all benchmark subdirectories.

using InteractiveUtils: versioninfo
using JSON: JSON
using PrettyTables: pretty_table

"""
    get_backend() -> String

Determine the benchmark backend from the `BENCHMARK_GROUP` environment variable.
If not set, auto-detects from available Reactant devices.
Returns one of "CPU", "CUDA", or "TPU".
"""
function get_backend()
    BENCHMARK_GROUP = get(ENV, "BENCHMARK_GROUP", nothing)

    if BENCHMARK_GROUP == "CUDA"
        Reactant.set_default_backend("gpu")
        @info "Running CUDA benchmarks" maxlog = 1
    elseif BENCHMARK_GROUP == "TPU"
        Reactant.set_default_backend("tpu")
        @info "Running TPU benchmarks" maxlog = 1
    elseif BENCHMARK_GROUP == "CPU"
        Reactant.set_default_backend("cpu")
        @info "Running CPU benchmarks" maxlog = 1
    else
        BENCHMARK_GROUP = String(split(string(first(Reactant.devices())), ":")[1])
        @info "Running $(BENCHMARK_GROUP) benchmarks" maxlog = 1
    end

    @assert BENCHMARK_GROUP in ("CPU", "CUDA", "TPU") "Unknown backend: $(BENCHMARK_GROUP)"
    return BENCHMARK_GROUP
end

"""
    save_results(results::Dict{String,Dict{String,Float64}},
                 results_dir::String, prefix::String,
                 backend::String)

Save benchmark results to a standardized JSON file.

- `results`: Dictionary mapping benchmark names to times (in seconds).
- `results_dir`: Directory to save the JSON file in (created if it doesn't exist).
- `prefix`: Filename prefix (e.g. "polybench", "misc", "nn").
- `backend`: Backend name (e.g. "CPU", "CUDA", "TPU").
"""
function save_results(
    results::Dict{String,Dict{String,Float64}},
    results_dir::String,
    prefix::String,
    backend::String,
)
    mkpath(results_dir)

    if isempty(prefix)
        benchmark_filename = string(backend, "benchmarks.json")
        tflops_filename = string(backend, "benchmarks_tflops.json")
    else
        benchmark_filename = string(prefix, "_", backend, "benchmarks.json")
        tflops_filename = string(prefix, "_", backend, "benchmarks_tflops.json")
    end

    benchmark_filepath = joinpath(results_dir, benchmark_filename)
    tflops_filepath = joinpath(results_dir, tflops_filename)

    standardized_results = Vector{Dict{String,Union{String,Float64}}}(
        undef, length(results["Runtime (s)"])
    )
    for (i, (k, v)) in enumerate(results["Runtime (s)"])
        standardized_results[i] = Dict("name" => k, "value" => v, "unit" => "s")
    end

    tflops_results = Vector{Dict{String,Union{String,Float64}}}(
        undef, length(results["TFLOP/s"])
    )
    for (i, (k, v)) in enumerate(results["TFLOP/s"])
        tflops_results[i] = Dict("name" => k, "value" => v, "unit" => "TFLOP/s")
    end

    open(benchmark_filepath, "w") do io
        JSON.json(io, standardized_results; pretty=true)
    end

    open(tflops_filepath, "w") do io
        JSON.json(io, tflops_results; pretty=true)
    end

    @info "Saved $(length(results["Runtime (s)"])) benchmark results to \
           $(benchmark_filepath) and $(tflops_filepath)"
    return benchmark_filepath, tflops_filepath
end

"""
    pretty_print_results(results::Dict, suite::String, backend::String)

Pretty-print benchmark results in a formatted table.
Splits benchmark names on "/" into columns for readability.
"""
function pretty_print_results(results::Dict, suite::String, backend::String)
    runtime_results = get(results, "Runtime (s)", Dict{String,Float64}())
    tflops_results = get(results, "TFLOP/s", Dict{String,Float64}())

    if isempty(runtime_results)
        @warn "No benchmark results to display for $(suite)/$(backend)"
        return nothing
    end

    sorted_keys = sort(collect(keys(runtime_results)))
    table = Matrix{Any}(undef, length(sorted_keys), 6)

    for (i, k) in enumerate(sorted_keys)
        parts = split(k, "/")
        # Fill in up to 4 columns from the benchmark name parts
        while length(parts) < 4
            push!(parts, "")
        end
        table[i, 1] = parts[1]
        table[i, 2] = length(parts) >= 2 ? parts[2] : ""
        table[i, 3] = length(parts) >= 3 ? parts[3] : ""
        table[i, 4] = length(parts) >= 4 ? join(parts[4:end], "/") : ""
        table[i, 5] = runtime_results[k]
        table[i, 6] = get(tflops_results, k, -1.0)
    end

    println()
    println("="^120)
    println("  Benchmark Results: $(suite) / $(backend)")
    println("="^120)

    pretty_table(
        table;
        alignment=[:l, :l, :l, :l, :c, :c],
        column_labels=["Benchmark", "Mode", "Backend", "Passes", "Time (s)", "TFLOP/s"],
        display_size=(-1, -1),
    )

    println("="^120)
    return nothing
end
