using Lux: Lux, reactant_device
using Printf: @sprintf
using Reactant: Reactant, @compile
using Enzyme: Enzyme, Const
using Random: Random

include("../utils.jl")

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
    if !haskey(results, "TFLOP/s")
        results["TFLOP/s"] = Dict{String,Float64}()
    end
    if !haskey(results, "Runtime (s)")
        results["Runtime (s)"] = Dict{String,Float64}()
    end

    prim_or_rev = fwd_or_bwd == "forward" ? "primal" : "reverse"
    full_benchmark_name = string(benchmark_name, "/", prim_or_rev, "/", backend, "/", tag)
    @assert !haskey(results["Runtime (s)"], full_benchmark_name) "Benchmark already \
                                                                  exists: \
                                                                  $(full_benchmark_name)"
    @assert !haskey(results["TFLOP/s"], full_benchmark_name) "Benchmark already exists: \
                                                              $(full_benchmark_name)"

    x, ps, st = general_lux_setup(model, x_dims)

    fn = if fwd_or_bwd == "forward"
        st = Lux.testmode(st)
        Lux.apply
    elseif fwd_or_bwd == "backward"
        simple_gradient
    else
        error("Unknown fwd_or_bwd: $(fwd_or_bwd)")
    end

    prof_result = Reactant.Profiler.profile_with_xprof(
        fn, model, x, ps, st; nrepeat=10, warmup=3, compile_options
    )
    results["Runtime (s)"][full_benchmark_name] =
        prof_result.profiling_result.runtime_ns / 1e9
    results["TFLOP/s"][full_benchmark_name] =
        if prof_result.profiling_result.flops_data === nothing
            -1
        else
            prof_result.profiling_result.flops_data.RawFlopsRate / 1e12
        end

    GC.gc(true)

    print_stmt = @sprintf(
        "%100s     :     %.5gs\t%.5g TFLOP/s",
        full_benchmark_name,
        results["Runtime (s)"][full_benchmark_name],
        results["TFLOP/s"][full_benchmark_name]
    )
    @info print_stmt

    return nothing
end
