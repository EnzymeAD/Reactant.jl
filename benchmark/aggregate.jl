using BenchmarkTools

const BACKENDS = ["CPU", "CUDA", "TPU"]

const CPU_Results = joinpath(dirname(@__FILE__), "results", "CPUbenchmarks.json")
@assert(ispath(CPU_Results))

const RESULTS = BenchmarkTools.load(CPU_Results)[1]
@assert RESULTS isa BenchmarkTools.BenchmarkGroup

for backend in BACKENDS[2:end]
    @info "Aggregating results for $(backend)"
    filename = string(backend, "benchmarks.json")
    filepath = joinpath(dirname(@__FILE__), "results", filename)
    if !ispath(filepath)
        @warn "No file found at path: $(filepath)"
    else
        backend_results = BenchmarkTools.load(filepath)[1]
        if backend_results isa BenchmarkTools.BenchmarkGroup
            # <benchmark name>/<forward or reverse>/<backend>/pass options>
            for benchmark in keys(RESULTS)
                for pass in keys(RESULTS[benchmark])
                    for pass_options in keys(backend_results[benchmark][pass][backend])
                        RESULTS[benchmark][pass][backend][pass_options] = backend_results[benchmark][pass][backend][pass_options]
                    end
                end
            end
        else
            @warn "Unexpected file format for file at path: $(filepath)"
        end
    end
end

BenchmarkTools.save(
    joinpath(dirname(@__FILE__), "results", "combinedbenchmarks.json"), RESULTS
)
