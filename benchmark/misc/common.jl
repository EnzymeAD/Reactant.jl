using Reactant: Reactant, @compile
using Chairmarks: @b
using Printf: @sprintf

include("../utils.jl")

struct BenchmarkConfiguration
    name::String
    compile_options::Union{Nothing,Reactant.CompileOptions}
    nrepeat::Int
end

function BenchmarkConfiguration(
    name::String;
    compile_options::Union{Nothing,Reactant.CompileOptions}=nothing,
    nrepeat::Int=25,
)
    return BenchmarkConfiguration(name, compile_options, nrepeat)
end

function run_benchmark!(
    results::Dict,
    backend::String,
    benchmark_name::String,
    fn::F,
    cpu_args::Tuple,
    ra_args::Tuple;
    skip_cpu::Bool=false,
    configs::Vector{BenchmarkConfiguration},
    benchmark_seconds::Float64=5.0,
    benchmark_samples::Int=100,
) where {F}
    if !haskey(results, "Runtime (s)")
        results["Runtime (s)"] = Dict{String,Float64}()
    end
    if !haskey(results, "TFLOP/s")
        results["TFLOP/s"] = Dict{String,Float64}()
    end

    # Run CPU/Julia benchmark if on CPU backend
    if backend == "CPU" && !skip_cpu
        full_benchmark_name = string(benchmark_name, "/CPU/Julia")

        # Warmup
        fn(cpu_args...)

        # Benchmark using Chairmarks
        bench = @b fn($(cpu_args)...) seconds = benchmark_seconds evals = 1 samples =
            benchmark_samples

        results["Runtime (s)"][full_benchmark_name] = bench.time
        results["TFLOP/s"][full_benchmark_name] = -1 # TODO: use LIKWID to get the TFLOP/s

        print_stmt = @sprintf "%100s     :     %.5gs" full_benchmark_name bench.time
        @info print_stmt
        GC.gc(true)
    end

    # Run Reactant profiling for each config
    for config in configs
        full_benchmark_name = string(benchmark_name, "/", backend, "/", config.name)

        prof_results = Reactant.Profiler.profile_with_xprof(
            fn, ra_args...; nrepeat=config.nrepeat, compile_options=config.compile_options
        )
        results["Runtime (s)"][full_benchmark_name] =
            prof_results.profiling_result.runtime_ns / 1e9
        results["TFLOP/s"][full_benchmark_name] =
            if prof_results.profiling_result.flops_data === nothing
                -1
            else
                prof_results.profiling_result.flops_data.RawFlopsRate / 1e12
            end

        print_stmt = @sprintf(
            "%100s     :     %.5gs\t%.5g TFLOP/s",
            full_benchmark_name,
            results["Runtime (s)"][full_benchmark_name],
            results["TFLOP/s"][full_benchmark_name],
        )
        @info print_stmt
        GC.gc(true)
    end

    return nothing
end
