using JSON3

const BACKENDS = ["CPU", "CUDA", "TPU"]

all_results = []
for backend in BACKENDS
    filename = string(backend, "benchmarks.json")
    filepath = joinpath(dirname(@__FILE__), "results", filename)
    if ispath(filepath)
        results = JSON3.read(read(filepath, String))
        append!(all_results, results)
    else
        @warn "No file found at path: $(filepath)"
    end
end

open(joinpath(dirname(@__FILE__), "results", "combinedbenchmarks.json"), "w") do io
    return JSON3.pretty(io, JSON3.write(all_results))
end
