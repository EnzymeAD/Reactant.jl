using BenchmarkTools: @benchmark
using Reactant, Enzyme, PrettyTables, Statistics

function simple_mse_loss(model, x, z, ps, st)
    y, _ = Lux.apply(model, x, ps, st)
    return MSELoss()(y, z)
end

function simple_mse_loss_gradient(model, x, z, ps, st)
    return Enzyme.gradient(
        Reverse, simple_mse_loss, Const(model), Const(x), Const(z), ps, Const(st)
    )
end

function benchmark_nn_primal(
    model, x, z, ps, st; disable_scatter_gather_bench=true, disable_pad_bench=true
)
    results = Vector{Tuple{String,String,Float64,Float64,Float64}}()

    # Only XLA
    compiled_fwd_xla = @compile compile_options = Reactant.DefaultXLACompileOptions(;
        sync=true
    ) simple_mse_loss(model, x, z, ps, st)
    bench = @benchmark $compiled_fwd_xla($model, $x, $z, $ps, $st) setup = (GC.gc(true))
    push!(results, ("Primal", "Only XLA", mean(bench).time, std(bench).time, 1.0))
    baseline = mean(bench).time

    # Default
    compiled_fwd = @compile compile_options = CompileOptions(;
        sync=true, no_nan=true, all_finite=true
    ) simple_mse_loss(model, x, z, ps, st)
    bench = @benchmark $compiled_fwd($model, $x, $z, $ps, $st) setup = (GC.gc(true))
    push!(
        results,
        ("Primal", "All", mean(bench).time, std(bench).time, mean(bench).time / baseline),
    )

    # Disable Scatter
    if disable_scatter_gather_bench
        compiled_fwd_no_scatter = @compile compile_options = CompileOptions(;
            disable_scatter_gather_optimization_passes=true,
            sync=true,
            no_nan=true,
            all_finite=true,
        ) simple_mse_loss(model, x, z, ps, st)
        bench = @benchmark $compiled_fwd_no_scatter($model, $x, $z, $ps, $st) setup = (GC.gc(
            true
        ))

        push!(
            results,
            (
                "Primal",
                "No Scatter/Gather Optimizations",
                mean(bench).time,
                std(bench).time,
                mean(bench).time / baseline,
            ),
        )
    end

    # Disable Pad
    if disable_pad_bench
        compiled_fwd_no_pad = @compile compile_options = CompileOptions(;
            disable_pad_optimization_passes=true, sync=true, no_nan=true, all_finite=true
        ) simple_mse_loss(model, x, z, ps, st)
        bench = @benchmark $compiled_fwd_no_pad($model, $x, $z, $ps, $st) setup = (GC.gc(
            true
        ))

        push!(
            results,
            (
                "Primal",
                "No Pad Optimizations",
                mean(bench).time,
                std(bench).time,
                mean(bench).time / baseline,
            ),
        )
    end

    # Disable Scatter and Pad
    if disable_scatter_gather_bench && disable_pad_bench
        compiled_fwd_no_scatter_pad = @compile compile_options = CompileOptions(;
            disable_scatter_gather_optimization_passes=true,
            disable_pad_optimization_passes=true,
            sync=true,
            no_nan=true,
            all_finite=true,
        ) simple_mse_loss(model, x, z, ps, st)
        bench = @benchmark $compiled_fwd_no_scatter_pad($model, $x, $z, $ps, $st) setup = (GC.gc(
            true
        ))

        push!(
            results,
            (
                "Primal",
                "No Scatter/Gather and Pad Optimizations",
                mean(bench).time,
                std(bench).time,
                mean(bench).time / baseline,
            ),
        )
    end

    sort!(results; by=x -> x[3])
    return results
end

function benchmark_nn_gradient(model, x, z, ps, st; kwargs...)
    return vcat(
        [
            benchmark_nn_gradient_internal(model, x, z, ps, st, mode; kwargs...) for
            mode in [:all, :before_enzyme, :after_enzyme]
        ]...,
    )
end

function benchmark_nn_gradient_internal(
    model, x, z, ps, st, mode; disable_scatter_gather_bench=true, disable_pad_bench=true
)
    @info "Benchmarking gradient with mode: $(Meta.quot(mode))"

    results = Vector{Tuple{String,String,Float64,Float64,Float64}}()

    # Only XLA
    compiled_grad_xla = @compile compile_options = Reactant.DefaultXLACompileOptions(;
        sync=true
    ) simple_mse_loss_gradient(model, x, z, ps, st)
    bench = @benchmark $compiled_grad_xla($model, $x, $z, $ps, $st) setup = (GC.gc(true))
    push!(results, ("Gradient ($mode)", "Only XLA", mean(bench).time, std(bench).time, 1.0))
    baseline = mean(bench).time

    display(results[end])

    # Default
    compiled_grad = @compile compile_options = CompileOptions(;
        sync=true, no_nan=true, all_finite=true, optimization_passes=mode
    ) simple_mse_loss_gradient(model, x, z, ps, st)
    bench = @benchmark $compiled_grad($model, $x, $z, $ps, $st) setup = (GC.gc(true))
    push!(
        results,
        (
            "Gradient ($mode)",
            "All",
            mean(bench).time,
            std(bench).time,
            mean(bench).time / baseline,
        ),
    )

    display(results[end])

    # Disable Scatter
    if disable_scatter_gather_bench
        compiled_grad_no_scatter = @compile compile_options = CompileOptions(;
            disable_scatter_gather_optimization_passes=true,
            optimization_passes=mode,
            sync=true,
            no_nan=true,
            all_finite=true,
        ) simple_mse_loss_gradient(model, x, z, ps, st)
        bench = @benchmark $compiled_grad_no_scatter($model, $x, $z, $ps, $st) setup = (GC.gc(
            true
        ))

        push!(
            results,
            (
                "Gradient ($mode)",
                "No Scatter/Gather Optimizations",
                mean(bench).time,
                std(bench).time,
                mean(bench).time / baseline,
            ),
        )

        display(results[end])
    end

    # Disable Pad
    if disable_pad_bench
        compiled_grad_no_pad = @compile compile_options = CompileOptions(;
            disable_pad_optimization_passes=true,
            optimization_passes=mode,
            sync=true,
            no_nan=true,
            all_finite=true,
        ) simple_mse_loss_gradient(model, x, z, ps, st)
        bench = @benchmark $compiled_grad_no_pad($model, $x, $z, $ps, $st) setup = (GC.gc(
            true
        ))

        push!(
            results,
            (
                "Gradient ($mode)",
                "No Pad Optimizations",
                mean(bench).time,
                std(bench).time,
                mean(bench).time / baseline,
            ),
        )

        display(results[end])
    end

    # Disable Pad and Scatter
    if disable_scatter_gather_bench && disable_pad_bench
        compiled_grad_no_scatter_no_pad = @compile compile_options = CompileOptions(;
            disable_scatter_gather_optimization_passes=true,
            disable_pad_optimization_passes=true,
            optimization_passes=mode,
            sync=true,
            no_nan=true,
            all_finite=true,
        ) simple_mse_loss_gradient(model, x, z, ps, st)
        bench = @benchmark $compiled_grad_no_scatter_no_pad($model, $x, $z, $ps, $st) setup = (GC.gc(
            true
        ))

        push!(
            results,
            (
                "Gradient ($mode)",
                "No Scatter/Gather/Pad Optimizations",
                mean(bench).time,
                std(bench).time,
                mean(bench).time / baseline,
            ),
        )

        display(results[end])
    end

    sort!(results; by=x -> x[3])
    return results
end

function pretty_print_table(results)
    header = (
        ["Mode", "Optimization Passes", "Mean Time", "Std. Dev. Time", "Relative Timing"],
        ["", "", "s", "s", "Time / XLA Time"],
    )

    results = copy(results)
    results[:, 3] ./= 1e9
    results[:, 4] ./= 1e9

    hl_r = Highlighter((data, i, j) -> j == 5 && data[i, j] > 1.0, crayon"bold red")
    hl_g = Highlighter((data, i, j) -> j == 5 && data[i, j] < 1.0, crayon"bold green")
    display(
        pretty_table(
            results;
            header,
            header_crayon=crayon"yellow bold",
            highlighters=(hl_r, hl_g),
            tf=tf_unicode_rounded,
        ),
    )
    return nothing
end
