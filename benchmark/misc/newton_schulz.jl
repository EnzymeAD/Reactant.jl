using Reactant, LinearAlgebra
using Random: Random

include("common.jl")

# Generic implementation of Newton-Schulz Orthogonalization Iteration
function newton_schulz_muon(G; steps=12, eps=1e-7)
    T = eltype(G)
    a, b, c = (T(3.4445), T(-4.7750), T(2.0315))

    # Normalize
    X = G
    X = X ./ (norm(X) + T(eps))

    if size(G, 1) > size(G, 2)
        X = X'
    end

    @trace for _ in 1:steps
        A = X * X'
        B = b .* A .+ c .* (A * A)
        X = a .* X .+ B * X
    end

    if size(G, 1) > size(G, 2)
        X = X'
    end

    return X
end

function run_newton_schulz_benchmark!(results, backend)
    compile_modes = [
        (
            "StructuredTensors",
            CompileOptions(; disable_structured_tensors_detection_passes=false),
        ),
        (
            "StructuredTensors (Only Detection)",
            CompileOptions(;
                disable_structured_tensors_detection_passes=false,
                disable_structured_tensors_passes=true,
            ),
        ),
        ("Default", CompileOptions(; disable_structured_tensors_detection_passes=true)),
    ]

    # Using a set of sizes to show scaling
    for N in [256, 1024, 4096]
        benchmark_name = "NewtonSchulz [$(N) x $(N)]/primal"

        # Initialize Random Data on CPU
        rng = Random.default_rng()
        X = randn(rng, Float32, N, N)
        X ./= opnorm(X)

        # Move to Reactant Device
        x_ra = Reactant.to_rarray(X)

        if backend == "CPU"
            full_benchmark_name = string(benchmark_name, "/CPU/Julia")

            # Warmup
            newton_schulz_muon(X)

            # Benchmark
            bench = @b newton_schulz_muon($X) seconds = 5 evals = 1 samples = 100

            results[full_benchmark_name] = bench.time

            print_stmt = @sprintf "%100s     :     %.5gs" full_benchmark_name bench.time
            @info print_stmt
            GC.gc(true)
        end

        for (tag, options) in compile_modes
            full_benchmark_name = string(benchmark_name, "/", backend, "/", tag)

            time = Reactant.Profiler.profile_with_xprof(
                newton_schulz_muon, x_ra; nrepeat=25, compile_options=options
            )
            time = time.profiling_result.runtime_ns / 1e9
            results[full_benchmark_name] = time

            print_stmt = @sprintf "%100s     :     %.5gs" full_benchmark_name time
            @info print_stmt
            GC.gc(true)
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    backend = get_backend()
    results = Dict()
    run_newton_schulz_benchmark!(results, backend)
end
