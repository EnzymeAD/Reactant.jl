function sumabs2first(model, x, ps, st)
    y, _ = Lux.apply(model, x, ps, st)
    return sum(abs2, y)
end

function simple_gradient(model, x, ps, st)
    return Enzyme.gradient(
        Enzyme.Reverse,
        sumabs2first,
        Enzyme.Const(model),
        Enzyme.Const(x),
        ps,
        Enzyme.Const(st),
    )
end

function general_lux_setup(model, x_dims)
    rng = Random.default_rng()  # don't use any other rng
    ps, st = Lux.setup(rng, model)
    x_dims === nothing && return ps, st
    x = randn(rng, Float32, x_dims)
    return x, ps, st
end

function setup_lux_benchmark!(
    suite::BenchmarkGroup,
    benchmark_name::String,
    backend::String,
    model,
    x_dims;
    disable_scatter_gather_bench=true,
    disable_pad_bench=true,
)
    common_opts = (; sync=true, no_nan=true, all_finite=true)

    # XXX: enable once we have finished the initial tests
    fwd_options = [
        ("XLA", Reactant.DefaultXLACompileOptions(; sync=true)),
        ("Default", Reactant.CompileOptions(; common_opts...)),
    ]

    bwd_options = [
        ("XLA", Reactant.DefaultXLACompileOptions(; sync=true)),
        [
            (
                "Default" * join(uppercasefirst.(split(string(pass), "_")), ""),
                Reactant.CompileOptions(; common_opts..., optimization_passes=pass),
            ) for pass in (:all, :before_enzyme, :after_enzyme)
        ]...,
    ]

    # if disable_scatter_gather_bench
    #     push!(
    #         fwd_options,
    #         (
    #             "DisableScatterGather",
    #             Reactant.CompileOptions(;
    #                 disable_scatter_gather_optimization_passes=true, common_opts...
    #             ),
    #         ),
    #     )
    #     append!(
    #         bwd_options,
    #         [
    #             (
    #                 "DisableScatterGather" *
    #                 join(uppercasefirst.(split(string(pass), "_")), ""),
    #                 Reactant.CompileOptions(;
    #                     disable_scatter_gather_optimization_passes=true,
    #                     optimization_passes=pass,
    #                     common_opts...,
    #                 ),
    #             ) for pass in (:all, :before_enzyme, :after_enzyme)
    #         ],
    #     )
    # end

    # if disable_pad_bench
    #     push!(
    #         fwd_options,
    #         (
    #             "DisablePad",
    #             Reactant.CompileOptions(;
    #                 disable_pad_optimization_passes=true, common_opts...
    #             ),
    #         ),
    #     )
    #     append!(
    #         bwd_options,
    #         [
    #             (
    #                 "DisablePad" * join(uppercasefirst.(split(string(pass), "_")), ""),
    #                 Reactant.CompileOptions(;
    #                     disable_pad_optimization_passes=true, optimization_passes=pass,
    #                     common_opts...,
    #                 ),
    #             ) for pass in (:all, :before_enzyme, :after_enzyme)
    #         ],
    #     )
    # end

    # if disable_scatter_gather_bench && disable_pad_bench
    #     push!(
    #         fwd_options,
    #         (
    #             "DisableScatterGatherPad",
    #             Reactant.CompileOptions(;
    #                 disable_scatter_gather_optimization_passes=true,
    #                 disable_pad_optimization_passes=true,
    #                 common_opts...,
    #             ),
    #         ),
    #     )
    #     append!(
    #         bwd_options,
    #         [
    #             (
    #                 "DisableScatterGatherPad" *
    #                 join(uppercasefirst.(split(string(pass), "_")), ""),
    #                 Reactant.CompileOptions(;
    #                     disable_scatter_gather_optimization_passes=true,
    #                     disable_pad_optimization_passes=true,
    #                     optimization_passes=pass,
    #                     common_opts...,
    #                 ),
    #             ) for pass in (:all, :before_enzyme, :after_enzyme)
    #         ],
    #     )
    # end

    for (tag, compile_options) in fwd_options
        add_benchmark!(
            suite, benchmark_name, backend, "forward", tag, compile_options, model, x_dims
        )
    end

    for (tag, compile_options) in bwd_options
        add_benchmark!(
            suite, benchmark_name, backend, "backward", tag, compile_options, model, x_dims
        )
    end

    return nothing
end

function add_benchmark!(
    suite::BenchmarkGroup,
    benchmark_name::String,
    backend::String,
    fwd_or_bwd::String,
    tag::String,
    compile_options,
    model,
    x_dims,
)
    if fwd_or_bwd == "forward"
        suite[benchmark_name][fwd_or_bwd][backend][tag] = @benchmarkable begin
            compiled_fwd($model, x_ra, ps_ra, st_test_ra)
        end setup = begin
            GC.gc(true)
            x, ps, st = general_lux_setup($model, $x_dims)
            st_test = Lux.testmode(st)
            x_ra, ps_ra, st_test_ra = Reactant.to_rarray((x, ps, st_test))
            compiled_fwd = @compile compile_options = $compile_options Lux.apply(
                $model, x_ra, ps_ra, st_test_ra
            )
            GC.gc(true)
        end
    else
        suite[benchmark_name][fwd_or_bwd][backend][tag] = @benchmarkable begin
            compiled_fwd($model, x_ra, ps_ra, st_ra)
        end setup = begin
            GC.gc(true)
            x, ps, st = general_lux_setup($model, $x_dims)
            x_ra, ps_ra, st_ra = Reactant.to_rarray((x, ps, st))
            compiled_fwd = @compile compile_options = $compile_options simple_gradient(
                $model, x_ra, ps_ra, st_ra
            )
            GC.gc(true)
        end
    end
    return nothing
end
