function simple_mse_loss(model, x, z, ps, st)
    y, _ = Lux.apply(model, x, ps, st)
    return MSELoss()(y, z)
end

function simple_mse_loss_gradient(model, x, z, ps, st)
    return Enzyme.gradient(
        Enzyme.Reverse, simple_mse_loss, Const(model), Const(x), Const(z), ps, Const(st)
    )
end

function general_lux_setup(model, x_dims)
    rng = Random.default_rng()  # don't use any other rng
    ps, st = Lux.setup(rng, model)
    x_dims === nothing && return ps, st
    x = randn(rng, Float32, x_dims)
    return x, ps, st
end

function setup_lux_forward_pass_benchmark!(
    suite::BenchmarkGroup,
    benchmark_name::String,
    backend::String,
    model,
    x_dims;
    disable_scatter_gather_bench=true,
    disable_pad_bench=true,
)
    common_opts = (; sync=true, no_nan=true, all_finite=true)

    options = [
        ("XLA", Reactant.DefaultXLACompileOptions(; sync=true)),
        ("Default", Reactant.CompileOptions(; common_opts...)),
    ]

    if disable_scatter_gather_bench
        push!(
            options,
            (
                "DisableScatterGather",
                Reactant.CompileOptions(;
                    disable_scatter_gather_optimization_passes=true, common_opts...
                ),
            ),
        )
    end

    if disable_pad_bench
        push!(
            options,
            (
                "DisablePad",
                Reactant.CompileOptions(;
                    disable_pad_optimization_passes=true, common_opts...
                ),
            ),
        )
    end

    if disable_scatter_gather_bench && disable_pad_bench
        push!(
            options,
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

    for (tag, compile_options) in options
        add_benchmark!(suite, benchmark_name, backend, tag, compile_options, model, x_dims)
    end

    return nothing
end

function add_benchmark!(
    suite::BenchmarkGroup,
    benchmark_name::String,
    backend::String,
    tag::String,
    compile_options,
    model,
    x_dims,
)
    suite[benchmark_name]["forward"][backend][tag] = @benchmarkable begin
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
    return nothing
end
