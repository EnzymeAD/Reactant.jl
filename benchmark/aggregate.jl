using JSON: JSON

const BACKENDS = ["CPU", "CUDA", "TPU"]
const SUITES = filter(readdir(@__DIR__)) do dir
    path = joinpath(@__DIR__, dir)
    return isdir(path) && ispath(joinpath(path, "runbenchmarks.jl"))
end

all_results = []
all_results_tflops = []

for backend in BACKENDS
    for suite in SUITES
        for (tag, arr) in
            [("benchmarks.json", all_results), ("benchmarks_tflops.json", all_results_tflops)]
            filename = string(suite, "_", backend, tag)
            filepath = joinpath(dirname(@__FILE__), "results", filename)
            if ispath(filepath)
                results = JSON.parsefile(filepath)
                append!(arr, results)
                @info "Loaded $(length(results)) results from $(filepath)"
            else
                @warn "No file found at path: $(filepath)"
            end
        end
    end
end

@info "Total results: $(length(all_results)) runtime, $(length(all_results_tflops)) TFLOP/s"

open(joinpath(dirname(@__FILE__), "results", "combinedbenchmarks.json"), "w") do io
    return JSON.json(io, all_results; pretty=true)
end

open(joinpath(dirname(@__FILE__), "results", "combinedbenchmarks_tflops.json"), "w") do io
    return JSON.json(io, all_results_tflops; pretty=true)
end
