using BenchmarkTools: @benchmark
using Reactant, Enzyme, PrettyTables, Statistics
using CairoMakie, AlgebraOfGraphics, CSV, DataFrames, Dates
const AoG = AlgebraOfGraphics

AoG.set_aog_theme!()

function simple_mse_loss(model, x, z, ps, st)
    y, _ = Lux.apply(model, x, ps, st)
    return MSELoss()(y, z)
end

function simple_mse_loss_gradient(model, x, z, ps, st)
    return Enzyme.gradient(
        Enzyme.Reverse, simple_mse_loss, Const(model), Const(x), Const(z), ps, Const(st)
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
                "No Scatter/Gather/Pad Optimizations",
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

function save_benchmark_results(
    results::Matrix,
    tag;
    savedir=tempname(; cleanup=false),
    device_tag=lowercase(
        replace(Reactant.XLA.device_kind(Reactant.devices()[1]), " " => "_")
    ),
    plot_title="",
)
    IN_VSCODE = isdefined(Main, :VSCodeServer)

    short_forms = Dict(
        "All" => "All",
        "Only XLA" => "Only XLA",
        "No Pad Optimizations" => "- Pad Opt",
        "No Scatter/Gather Optimizations" => "- S.G. Opt",
        "No Scatter/Gather/Pad Optimizations" => "- S.G. + Pad Opt",
        "No Scatter/Gather and Pad Optimizations" => "- S.G. + Pad Opt",
    )

    mkpath(savedir)
    file_name_base = "$(tag)_$(device_tag)_$(Dates.format(now(), "yyyy_mm_dd_HH_MM_SS"))"

    df = DataFrame(
        results,
        ["Mode", "Optimization Passes", "Mean Time", "Std. Dev. Time", "Relative Timing"],
    )

    csv_results_file_name = joinpath(savedir, "$(file_name_base).csv")
    # CSV.write(csv_results_file_name, df) # XXX: enable

    @info "Saving timings to $(csv_results_file_name)"

    df[!, "μ - σ"] = df[!, "Mean Time"] .- df[!, "Std. Dev. Time"]
    df[!, "μ + σ"] = df[!, "Mean Time"] .+ df[!, "Std. Dev. Time"]

    fig = Figure(; size=(1000, 500), title="Reactant XLA Timings")
    draw!(
        fig,
        (
            data(df) *
            mapping(
                "Optimization Passes" => x -> short_forms[x],
                "Mean Time";
                color="Optimization Passes" => "",
                col="Mode",
            ) *
            visual(BarPlot; strokewidth=2)
        ) + (
            data(df) *
            mapping(
                "Optimization Passes" => x -> short_forms[x], "μ - σ", "μ + σ"; col="Mode"
            ) *
            visual(Rangebars; linewidth=2)
        ),
        scales(; Color=(; palette=:tab10));
        axis=(; xticklabelrotation=π / 3),
    )

    if !isempty(plot_title)
        Label(
            fig[begin - 1, :],
            plot_title;
            tellwidth=false,
            font=:bold,
            fontsize=1.15 * Makie.theme(fig.scene).fontsize[],
            halign=:center,
        )
    end

    IN_VSCODE && display(fig)

    plots_file_name = joinpath(savedir, "$(file_name_base).pdf")
    save(plots_file_name, fig)

    @info "Saving plots to $(plots_file_name)"

    return nothing
end
