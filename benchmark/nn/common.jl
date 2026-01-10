using Lux: Lux, reactant_device
using Printf: @sprintf
using Reactant: Reactant, @compile
using Enzyme: Enzyme, Const
using Random: Random

sumabs2first(model, x, ps, st) = sum(abs2, first(Lux.apply(model, x, ps, st)))

function simple_gradient(model, x, ps, st)
    return Enzyme.gradient(
        Enzyme.Reverse, sumabs2first, Const(model), Const(x), ps, Const(st)
    )
end

init_array(rng, dims::Tuple) = init_array.(Ref(rng), dims)
init_array(rng, dims::Integer) = randn(rng, Float32, dims)
init_array(rng, dims::Dims) = randn(rng, Float32, dims)

function general_lux_setup(model, x_dims)
    rng = Random.default_rng()  # don't use any other rng
    Random.seed!(rng, 0)
    xdev = reactant_device(; force=true)
    ps, st = xdev(Lux.setup(rng, model))
    x_dims === nothing && return ps, st
    x = xdev(init_array(rng, x_dims))
    return x, ps, st
end

function run_lux_benchmark!(
    results::Dict,
    benchmark_name::String,
    backend::String,
    model,
    x_dims;
    disable_scatter_gather_bench=true,
    disable_bwd_scatter_gather_bench=true,
    disable_pad_bench=true,
    disable_bwd_pad_bench=true,
    disable_transpose_bench=true,
    disable_bwd_transpose_bench=true,
    bwd_enzyme_pass_options=(:all, :before_enzyme, :after_enzyme),
)
    common_opts = (; no_nan=true, all_finite=true)

    fwd_options = [
        ("NoOpt", Reactant.DefaultXLACompileOptions()),
        ("Default", Reactant.CompileOptions(; common_opts...)),
    ]

    bwd_options = [
        ("NoOpt", Reactant.DefaultXLACompileOptions()),
        [
            (
                "Default" * join(uppercasefirst.(split(string(pass), "_")), ""),
                Reactant.CompileOptions(; common_opts..., optimization_passes=pass),
            ) for pass in bwd_enzyme_pass_options
        ]...,
    ]

    if !disable_transpose_bench
        push!(
            fwd_options,
            (
                "DisableTransposeReshape",
                Reactant.CompileOptions(;
                    transpose_propagate=:none, reshape_propagate=:none, common_opts...
                ),
            ),
        )
    end

    if !disable_bwd_transpose_bench
        append!(
            bwd_options,
            [
                (
                    "DisableTransposeReshape" *
                    join(uppercasefirst.(split(string(pass), "_")), ""),
                    Reactant.CompileOptions(;
                        transpose_propagate=:none,
                        reshape_propagate=:none,
                        optimization_passes=pass,
                        common_opts...,
                    ),
                ) for pass in bwd_enzyme_pass_options
            ],
        )
    end

    if !disable_scatter_gather_bench
        push!(
            fwd_options,
            (
                "DisableScatterGather",
                Reactant.CompileOptions(;
                    disable_scatter_gather_optimization_passes=true, common_opts...
                ),
            ),
        )
    end

    if !disable_bwd_scatter_gather_bench
        append!(
            bwd_options,
            [
                (
                    "DisableScatterGather" *
                    join(uppercasefirst.(split(string(pass), "_")), ""),
                    Reactant.CompileOptions(;
                        disable_scatter_gather_optimization_passes=true,
                        optimization_passes=pass,
                        common_opts...,
                    ),
                ) for pass in bwd_enzyme_pass_options
            ],
        )
    end

    if !disable_pad_bench
        push!(
            fwd_options,
            (
                "DisablePad",
                Reactant.CompileOptions(;
                    disable_pad_optimization_passes=true, common_opts...
                ),
            ),
        )
    end

    if !disable_bwd_pad_bench
        append!(
            bwd_options,
            [
                (
                    "DisablePad" * join(uppercasefirst.(split(string(pass), "_")), ""),
                    Reactant.CompileOptions(;
                        disable_pad_optimization_passes=true,
                        optimization_passes=pass,
                        common_opts...,
                    ),
                ) for pass in bwd_enzyme_pass_options
            ],
        )
    end

    if !disable_scatter_gather_bench && !disable_pad_bench
        push!(
            fwd_options,
            (
                "DisableScatterGatherPad",
                Reactant.CompileOptions(;
                    disable_scatter_gather_optimization_passes=true,
                    disable_pad_optimization_passes=true,
                    common_opts...,
                ),
            ),
        )
    end

    if !disable_bwd_scatter_gather_bench && !disable_bwd_pad_bench
        append!(
            bwd_options,
            [
                (
                    "DisableScatterGatherPad" *
                    join(uppercasefirst.(split(string(pass), "_")), ""),
                    Reactant.CompileOptions(;
                        disable_scatter_gather_optimization_passes=true,
                        disable_pad_optimization_passes=true,
                        optimization_passes=pass,
                        common_opts...,
                    ),
                ) for pass in bwd_enzyme_pass_options
            ],
        )
    end

    for (tag, compile_options) in fwd_options
        run_benchmark!(
            results, benchmark_name, backend, "forward", tag, compile_options, model, x_dims
        )
    end

    for (tag, compile_options) in bwd_options
        run_benchmark!(
            results,
            benchmark_name,
            backend,
            "backward",
            tag,
            compile_options,
            model,
            x_dims,
        )
    end

    return nothing
end

function run_benchmark!(
    results::Dict,
    benchmark_name::String,
    backend::String,
    fwd_or_bwd::String,
    tag::String,
    compile_options,
    model,
    x_dims,
)
    prim_or_rev = fwd_or_bwd == "forward" ? "primal" : "reverse"
    full_benchmark_name = string(benchmark_name, "/", prim_or_rev, "/", backend, "/", tag)
    @assert !haskey(results, full_benchmark_name) "Benchmark already exists: \
                                                   $(full_benchmark_name)"

    if fwd_or_bwd == "forward"
        x, ps, st = general_lux_setup(model, x_dims)
        time = Reactant.Profiler.profile_with_xprof(
            Lux.apply, model, x, ps, Lux.testmode(st); nrepeat=10, warmup=3, compile_options
        )
        time = time.profiling_result.runtime_ns / 1e9
        results[full_benchmark_name] = time
        GC.gc(true)
    elseif fwd_or_bwd == "backward"
        x, ps, st = general_lux_setup(model, x_dims)
        time = Reactant.Profiler.profile_with_xprof(
            simple_gradient, model, x, ps, st; nrepeat=10, warmup=3, compile_options
        )
        time = time.profiling_result.runtime_ns / 1e9
        results[full_benchmark_name] = time
        GC.gc(true)
    else
        @error "Unknown fwd_or_bwd: $(fwd_or_bwd)"
    end

    print_stmt = @sprintf "%100s     :     %.5gs" full_benchmark_name time
    @info print_stmt

    return nothing
end

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
