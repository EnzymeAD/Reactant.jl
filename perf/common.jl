using BenchmarkTools: @belapsed
using Reactant, Enzyme, PrettyTables, Statistics

function simple_mse_loss(model, x, ps, st)
    y, _ = Lux.apply(model, x, ps, st)
    return sum(abs2, y)
end

function benchmark_nn_primal(
    model, x, ps, st; disable_scatter_gather_bench=true, disable_pad_bench=true
)
    results = Vector{Tuple{String,String,Float64,Float64,Float64}}()

    # Only XLA
    compiled_fwd_xla = @compile sync = true compile_options = Reactant.DefaultXLACompileOptions() simple_mse_loss(
        model, x, ps, st
    )
    bench = @benchmark $compiled_fwd_xla($model, $x, $ps, $st)
    push!(results, ("Primal", "Only XLA", median(bench).time, std(bench).time, 1.0))
    baseline = median(bench).time

    # Default
    compiled_fwd = @compile sync = true simple_mse_loss(model, x, ps, st)
    bench = @benchmark $compiled_fwd($model, $x, $ps, $st)
    push!(
        results,
        (
            "Primal",
            "All",
            median(bench).time,
            std(bench).time,
            median(bench).time / baseline,
        ),
    )

    # Disable Scatter
    if disable_scatter_gather_bench
        compiled_fwd_no_scatter = @compile sync = true compile_options = CompileOptions(;
            disable_scatter_gather_optimization_passes=true
        ) simple_mse_loss(model, x, ps, st)
        bench = @benchmark $compiled_fwd_no_scatter($model, $x, $ps, $st)

        push!(
            results,
            (
                "Primal",
                "No Scatter/Gather Optimizations",
                median(bench).time,
                std(bench).time,
                median(bench).time / baseline,
            ),
        )
    end

    # Disable Pad
    if disable_pad_bench
        compiled_fwd_no_pad = @compile sync = true compile_options = CompileOptions(;
            disable_pad_optimization_passes=true
        ) simple_mse_loss(model, x, ps, st)
        bench = @benchmark $compiled_fwd_no_pad($model, $x, $ps, $st)

        push!(
            results,
            (
                "Primal",
                "No Pad Optimizations",
                median(bench).time,
                std(bench).time,
                median(bench).time / baseline,
            ),
        )
    end

    # Disable Scatter and Pad
    if disable_scatter_gather_bench && disable_pad_bench
        compiled_fwd_no_scatter_pad = @compile sync = true compile_options = CompileOptions(;
            disable_scatter_gather_optimization_passes=true,
            disable_pad_optimization_passes=true,
        ) simple_mse_loss(model, x, ps, st)
        bench = @benchmark $compiled_fwd_no_scatter_pad($model, $x, $ps, $st)

        push!(
            results,
            (
                "Primal",
                "No Scatter/Gather and Pad Optimizations",
                median(bench).time,
                std(bench).time,
                median(bench).time / baseline,
            ),
        )
    end

    sort!(results; by=x -> x[3])
    return results
end

function pretty_print_table(results)
    header = (
        ["Mode", "Optimization Passes", "Median Time", "Std. Dev. Time", "Relative Timing"],
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
