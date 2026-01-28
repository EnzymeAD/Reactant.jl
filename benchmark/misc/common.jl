using Reactant: Reactant, @compile
using Chairmarks: @b
using Printf: @sprintf

function get_backend()
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
    return BENCHMARK_GROUP
end

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
    results,
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
    # Run CPU/Julia benchmark if on CPU backend
    if backend == "CPU" && !skip_cpu
        full_benchmark_name = string(benchmark_name, "/CPU/Julia")

        # Warmup
        fn(cpu_args...)

        # Benchmark using Chairmarks
        bench = @b fn($(cpu_args)...) seconds = benchmark_seconds evals = 1 samples =
            benchmark_samples

        results[full_benchmark_name] = bench.time

        print_stmt = @sprintf "%100s     :     %.5gs" full_benchmark_name bench.time
        @info print_stmt
        GC.gc(true)
    end

    # Run Reactant profiling for each config
    for config in configs
        full_benchmark_name = string(benchmark_name, "/", backend, "/", config.name)

        time = Reactant.Profiler.profile_with_xprof(
            fn, ra_args...; nrepeat=config.nrepeat, compile_options=config.compile_options
        )
        time = time.profiling_result.runtime_ns / 1e9
        results[full_benchmark_name] = time

        print_stmt = @sprintf "%100s     :     %.5gs" full_benchmark_name time
        @info print_stmt
        GC.gc(true)
    end

    return nothing
end
