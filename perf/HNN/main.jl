using Lux,
    Random,
    Reactant,
    Enzyme,
    Zygote,
    BenchmarkTools,
    LuxCUDA,
    DataFrames,
    OrderedCollections,
    CSV,
    Comonicon

struct HamiltonianNN{E,M} <: AbstractLuxWrapperLayer{:model}
    model::M

    HamiltonianNN{E}(model::M) where {E,M} = new{E,M}(model)
end

function (hnn::HamiltonianNN{false})(x::AbstractArray, ps, st)
    model = StatefulLuxLayer{true}(hnn.model, ps, st)
    ∂x = only(Zygote.gradient(sum ∘ model, x))
    n = size(x, ndims(x) - 1) ÷ 2
    y = cat(
        selectdim(∂x, ndims(∂x) - 1, (n + 1):(2n)),
        selectdim(∂x, ndims(∂x) - 1, 1:n);
        dims=Val(ndims(∂x) - 1),
    )
    return y, model.st
end

function (hnn::HamiltonianNN{true})(x::AbstractArray, ps, st)
    ∂x = similar(x)
    model = StatefulLuxLayer{true}(hnn.model, ps, st)
    Enzyme.autodiff(Reverse, Const(sum ∘ model), Duplicated(x, ∂x))
    n = size(x, ndims(x) - 1) ÷ 2
    y = cat(
        selectdim(∂x, ndims(∂x) - 1, (n + 1):(2n)),
        selectdim(∂x, ndims(∂x) - 1, 1:n);
        dims=Val(ndims(∂x) - 1),
    )
    return y, model.st
end

function loss_fn(model, ps, st, x, y)
    pred, _ = model(x, ps, st)
    return MSELoss()(pred, y)
end

function ∇zygote_loss_fn(model, ps, st, x, y)
    _, dps, _, dx, _ = Zygote.gradient(loss_fn, model, ps, st, x, y)
    return dps, dx
end

function ∇enzyme_loss_fn(model, ps, st, x, y)
    _, dps, _, dx, _ = Enzyme.gradient(
        Reverse, loss_fn, Const(model), ps, Const(st), x, Const(y)
    )
    return dps, dx
end

function reclaim_fn(backend, reactant)
    if backend == "gpu" && !reactant
        CUDA.reclaim()
    end
    GC.gc(true)
    return nothing
end

Comonicon.@main function main(; backend::String="gpu")
    @assert backend in ("cpu", "gpu")

    Reactant.set_default_backend(backend)
    filename = joinpath(@__DIR__, "results_$(backend).csv")

    @info "Using backend" backend

    cdev = cpu_device()
    gdev = backend == "gpu" ? gpu_device(; force=true) : cdev
    xdev = reactant_device(; force=true)

    df = DataFrame(
        OrderedDict(
            "Kind" => [],
            "Fwd Vanilla" => [],
            "Fwd Reactant" => [],
            "Fwd Reactant SpeedUp" => [],
            "Bwd Zygote" => [],
            "Bwd Reactant" => [],
            "Bwd Reactant SpeedUp" => [],
        ),
    )

    mlp = Chain(
        Dense(32, 128, gelu),
        Dense(128, 128, gelu),
        Dense(128, 128, gelu),
        Dense(128, 128, gelu),
        Dense(128, 1),
    )

    model_enz = HamiltonianNN{true}(mlp)
    model_zyg = HamiltonianNN{false}(mlp)

    ps, st = Lux.setup(Random.default_rng(), model_enz)

    x = randn(Float32, 32, 1024)
    y = randn(Float32, 32, 1024)

    x_gdev = gdev(x)
    y_gdev = gdev(y)
    x_xdev = xdev(x)
    y_xdev = xdev(y)

    ps_gdev, st_gdev = gdev((ps, st))
    ps_xdev, st_xdev = xdev((ps, st))

    @info "Compiling Forward Functions"
    lfn_compiled = @compile sync = true loss_fn(model_enz, ps_xdev, st_xdev, x_xdev, y_xdev)

    @info "Running Forward Benchmarks"

    t_gdev = @belapsed CUDA.@sync(loss_fn($model_zyg, $ps_gdev, $st_gdev, $x_gdev, $y_gdev)) setup = (reclaim_fn(
        $backend, false
    ))

    t_xdev = @belapsed $lfn_compiled($model_enz, $ps_xdev, $st_xdev, $x_xdev, $y_xdev) setup = (reclaim_fn(
        $backend, true
    ))

    @info "Forward Benchmarks" t_gdev t_xdev

    @info "Compiling Backward Functions"
    grad_fn_compiled = @compile sync = true ∇enzyme_loss_fn(
        model_enz, ps_xdev, st_xdev, x_xdev, y_xdev
    )

    @info "Running Backward Benchmarks"

    t_rev_gdev = @belapsed CUDA.@sync(
        ∇zygote_loss_fn($model_zyg, $ps_gdev, $st_gdev, $x_gdev, $y_gdev)
    ) setup = (reclaim_fn($backend, false))

    t_rev_xdev = @belapsed $grad_fn_compiled(
        $model_enz, $ps_xdev, $st_xdev, $x_xdev, $y_xdev
    ) setup = (reclaim_fn($backend, true))

    @info "Backward Benchmarks" t_rev_gdev t_rev_xdev

    push!(
        df,
        [
            "HNN",
            t_gdev,
            t_xdev,
            t_gdev / t_xdev,
            t_rev_gdev,
            t_rev_xdev,
            t_rev_gdev / t_rev_xdev,
        ],
    )

    display(df)
    CSV.write(filename, df)

    @info "Results saved to $filename"
    return nothing
end
