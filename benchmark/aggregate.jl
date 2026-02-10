using JSON3

const BACKENDS = ["CPU", "CUDA", "TPU"]

all_results = []
all_results_tflops = []
for backend in BACKENDS
    for (tag, arr) in
        (["benchmarks.json", all_results], ["benchmark_tflops.json", all_results_tflops])
        filename = string(backend, tag)
        filepath = joinpath(dirname(@__FILE__), "results", filename)
        if ispath(filepath)
            results = JSON3.read(read(filepath, String))
            append!(arr, results)
        else
            @warn "No file found at path: $(filepath)"
        end
    end
end

open(joinpath(dirname(@__FILE__), "results", "combinedbenchmarks.json"), "w") do io
    return JSON3.pretty(io, JSON3.write(all_results))
end

open(joinpath(dirname(@__FILE__), "results", "combinedbenchmark_tflops.json"), "w") do io
    return JSON3.pretty(io, JSON3.write(all_results_tflops))
end
