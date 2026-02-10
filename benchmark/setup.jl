# Benchmark Orchestrator
# This script locates subdirectories with runbenchmarks.jl, runs each in a separate
# Julia process, and aggregates the results from saved JSON files.

using JSON3: JSON3

"""
    find_benchmark_dirs(base_dir::String) -> Vector{String}

Automatically find all subdirectories containing a `runbenchmarks.jl` file.
"""
function find_benchmark_dirs(base_dir::String)
    dirs = String[]
    for entry in readdir(base_dir; join=true)
        if isdir(entry)
            runbenchmarks_path = joinpath(entry, "runbenchmarks.jl")
            project_path = joinpath(entry, "Project.toml")
            if isfile(runbenchmarks_path) && isfile(project_path)
                push!(dirs, entry)
            end
        end
    end
    return sort(dirs)  # Sort for consistent ordering
end

"""
    instantiate_project(benchmark_dir::String)

Instantiate the Julia project in the given directory.
"""
function instantiate_project(benchmark_dir::String)
    benchmark_name = basename(benchmark_dir)
    @info "Instantiating project for $(benchmark_name)..."

    # Run Pkg.instantiate in a subprocess to avoid polluting the current environment
    cmd = Cmd([
        Base.julia_cmd().exec...,
        "--project=$(benchmark_dir)",
        "-e",
        "using Pkg; Pkg.instantiate()",
    ])

    run(pipeline(cmd; stdout=devnull, stderr=stderr); wait=true)
    @info "Successfully instantiated project for $(benchmark_name)"
    return nothing
end

"""
    run_benchmark_subprocess(benchmark_dir::String, backend::String)

Run the benchmarks in a subdirectory in a separate Julia process.
Results are saved to the subdirectory's results/ folder.
"""
function run_benchmark_subprocess(benchmark_dir::String, backend::String)
    runbenchmarks_path = joinpath(benchmark_dir, "runbenchmarks.jl")
    benchmark_name = basename(benchmark_dir)

    @info "Starting benchmarks for $(benchmark_name)..."

    # Build the command
    cmd = Cmd(
        Cmd([
            Base.julia_cmd().exec...,
            "--color=yes",
            "--project=$(benchmark_dir)",
            "--threads=$(Threads.nthreads())",
            runbenchmarks_path,
        ]);
        env=merge(ENV, Dict("BENCHMARK_GROUP" => backend)),
    )

    # Run the process - output goes to console, results are saved to files
    run(cmd; wait=true)

    @info "Completed benchmarks for $(benchmark_name)"
    return nothing
end

"""
    load_results_from_dir(benchmark_dir::String, backend::String) -> Vector{Dict}

Load benchmark results from a subdirectory's results folder.
"""
function load_results_from_dir(benchmark_dir::String, backend::String)
    results_dir = joinpath(benchmark_dir, "results")
    benchmark_name = basename(benchmark_dir)
    all_results = Dict{String,Any}[]

    if !isdir(results_dir)
        @warn "No results directory found for $(benchmark_name)"
        return all_results
    end

    for filename in readdir(results_dir)
        # Only load files matching the current backend
        if endswith(filename, ".json") && contains(filename, backend)
            filepath = joinpath(results_dir, filename)
            try
                results = JSON3.read(read(filepath, String), Vector{Dict{String,Any}})
                append!(all_results, results)
                @info "Loaded $(length(results)) results from $(filepath)"
            catch e
                @error "Failed to load results from $(filepath)" exception = e
            end
        end
    end

    return all_results
end

"""
    run_all_benchmarks(backend::String) -> Dict{String,Float64}

Run all benchmark subdirectories in separate processes and aggregate results.
"""
function run_all_benchmarks(backend::String)
    base_dir = dirname(@__FILE__)
    benchmark_dirs = find_benchmark_dirs(base_dir)

    @info "Found $(length(benchmark_dirs)) benchmark directories: $(basename.(benchmark_dirs))"

    # Instantiate all projects first
    for dir in benchmark_dirs
        instantiate_project(dir)
    end

    # Run all benchmarks
    for dir in benchmark_dirs
        run_benchmark_subprocess(dir, backend)
    end

    # Aggregate results from saved files
    all_results = Dict{String,Float64}()

    for dir in benchmark_dirs
        results = load_results_from_dir(dir, backend)

        # Merge results
        for result in results
            name = result["name"]
            value = result["value"]
            if haskey(all_results, name)
                @warn "Duplicate benchmark name" name
            end
            all_results[name] = value
        end
    end

    @info "Aggregated $(length(all_results)) total benchmark results"

    return all_results
end

"""
    aggregate_saved_results(base_dir::String) -> Vector{Dict}

Aggregate results from all subdirectory result files.
"""
function aggregate_saved_results(base_dir::String)
    all_results = Dict{String,Any}[]

    # Find all benchmark directories
    benchmark_dirs = find_benchmark_dirs(base_dir)

    for dir in benchmark_dirs
        results_dir = joinpath(dir, "results")
        if !isdir(results_dir)
            continue
        end

        for filename in readdir(results_dir)
            if endswith(filename, ".json")
                filepath = joinpath(results_dir, filename)
                try
                    results = JSON3.read(read(filepath, String), Vector{Dict{String,Any}})
                    append!(all_results, results)
                    @info "Loaded $(length(results)) results from $(filepath)"
                catch e
                    @error "Failed to load results from $(filepath)" exception = e
                end
            end
        end
    end

    return all_results
end
