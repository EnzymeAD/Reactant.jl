# This has been adapted from https://github.com/vpuri3/KolmogorovArnold.jl/blob/38616fc66b3c5c1550afa7c718a0629608def19b/examples/eg3.jl

using KolmogorovArnold,
    Lux,
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

    x = randn(Float32, 1, 1024)
    x_gdev = gdev(x)
    x_xdev = xdev(x)

    y_gdev = x_gdev .^ 2
    y_xdev = x_xdev .^ 2

    wM = 128
    wK = 40
    G = 10

    mlp = Chain(Dense(1, wM, tanh), Dense(wM, wK, tanh), Dense(wK, 1))

    basis_func = rbf
    normalizer = softsign

    kan1 = Chain(
        KDense(1, wK, G; use_base_act=true, basis_func, normalizer),
        KDense(wK, wK, G; use_base_act=true, basis_func, normalizer),
        KDense(wK, 1, G; use_base_act=true, basis_func, normalizer),
    )

    kan2 = Chain(
        KDense(1, wK, G; use_base_act=false, basis_func, normalizer),
        KDense(wK, wK, G; use_base_act=false, basis_func, normalizer),
        KDense(wK, 1, G; use_base_act=false, basis_func, normalizer),
    )

    ps_mlp, st_mlp = Lux.setup(Random.default_rng(), mlp)
    ps_kan1, st_kan1 = Lux.setup(Random.default_rng(), kan1)
    ps_kan2, st_kan2 = Lux.setup(Random.default_rng(), kan2)

    ps_mlp_gdev, st_mlp_gdev = gdev((ps_mlp, st_mlp))
    ps_kan1_gdev, st_kan1_gdev = gdev((ps_kan1, st_kan1))
    ps_kan2_gdev, st_kan2_gdev = gdev((ps_kan2, st_kan2))

    ps_mlp_xdev, st_mlp_xdev = xdev((ps_mlp, st_mlp))
    ps_kan1_xdev, st_kan1_xdev = xdev((ps_kan1, st_kan1))
    ps_kan2_xdev, st_kan2_xdev = xdev((ps_kan2, st_kan2))

    @info "Compiling Forward Functions"
    lfn_mlp_compiled = @compile sync = true loss_fn(
        mlp, ps_mlp_xdev, st_mlp_xdev, x_xdev, y_xdev
    )
    lfn_kan1_compiled = @compile sync = true loss_fn(
        kan1, ps_kan1_xdev, st_kan1_xdev, x_xdev, y_xdev
    )
    lfn_kan2_compiled = @compile sync = true loss_fn(
        kan2, ps_kan2_xdev, st_kan2_xdev, x_xdev, y_xdev
    )

    @info "Running Forward Benchmarks"

    tmlp_gdev = @belapsed CUDA.@sync(
        loss_fn($mlp, $ps_mlp_gdev, $st_mlp_gdev, $x_gdev, $y_gdev)
    ) setup = (reclaim_fn($backend, false))
    tkan1_gdev = @belapsed CUDA.@sync(
        loss_fn($kan1, $ps_kan1_gdev, $st_kan1_gdev, $x_gdev, $y_gdev)
    ) setup = (reclaim_fn($backend, false))
    tkan2_gdev = @belapsed CUDA.@sync(
        loss_fn($kan2, $ps_kan2_gdev, $st_kan2_gdev, $x_gdev, $y_gdev)
    ) setup = (reclaim_fn($backend, false))

    @info "Vanilla Forward Benchmarks" tmlp_gdev tkan1_gdev tkan2_gdev

    tmlp_xdev = @belapsed $lfn_mlp_compiled(
        $mlp, $ps_mlp_xdev, $st_mlp_xdev, $x_xdev, $y_xdev
    ) setup = (reclaim_fn($backend, true))
    tkan1_xdev = @belapsed $lfn_kan1_compiled(
        $kan1, $ps_kan1_xdev, $st_kan1_xdev, $x_xdev, $y_xdev
    ) setup = (reclaim_fn($backend, true))
    tkan2_xdev = @belapsed $lfn_kan2_compiled(
        $kan2, $ps_kan2_xdev, $st_kan2_xdev, $x_xdev, $y_xdev
    ) setup = (reclaim_fn($backend, true))

    @info "Reactant Forward Benchmarks" tmlp_xdev tkan1_xdev tkan2_xdev

    @info "Compiling Backward Functions"
    grad_fn_mlp_compiled = @compile sync = true ∇enzyme_loss_fn(
        mlp, ps_mlp_xdev, st_mlp_xdev, x_xdev, y_xdev
    )
    grad_fn_kan1_compiled = @compile sync = true ∇enzyme_loss_fn(
        kan1, ps_kan1_xdev, st_kan1_xdev, x_xdev, y_xdev
    )
    grad_fn_kan2_compiled = @compile sync = true ∇enzyme_loss_fn(
        kan2, ps_kan2_xdev, st_kan2_xdev, x_xdev, y_xdev
    )

    @info "Running Backward Benchmarks"

    tmlp_rev_gdev = @belapsed CUDA.@sync(
        ∇zygote_loss_fn($mlp, $ps_mlp_gdev, $st_mlp_gdev, $x_gdev, $y_gdev)
    ) setup = (reclaim_fn($backend, false))
    tkan1_rev_gdev = @belapsed CUDA.@sync(
        ∇zygote_loss_fn($kan1, $ps_kan1_gdev, $st_kan1_gdev, $x_gdev, $y_gdev)
    ) setup = (reclaim_fn($backend, false))
    tkan2_rev_gdev = @belapsed CUDA.@sync(
        ∇zygote_loss_fn($kan2, $ps_kan2_gdev, $st_kan2_gdev, $x_gdev, $y_gdev)
    ) setup = (reclaim_fn($backend, false))

    @info "Zygote Backward Benchmarks" tmlp_rev_gdev tkan1_rev_gdev tkan2_rev_gdev

    tmlp_rev_xdev = @belapsed $grad_fn_mlp_compiled(
        $mlp, $ps_mlp_xdev, $st_mlp_xdev, $x_xdev, $y_xdev
    ) setup = (reclaim_fn($backend, true))
    tkan1_rev_xdev = @belapsed $grad_fn_kan1_compiled(
        $kan1, $ps_kan1_xdev, $st_kan1_xdev, $x_xdev, $y_xdev
    ) setup = (reclaim_fn($backend, true))
    tkan2_rev_xdev = @belapsed $grad_fn_kan2_compiled(
        $kan2, $ps_kan2_xdev, $st_kan2_xdev, $x_xdev, $y_xdev
    ) setup = (reclaim_fn($backend, true))

    @info "Reactant Backward Benchmarks" tmlp_rev_xdev tkan1_rev_xdev tkan2_rev_xdev

    push!(
        df,
        [
            "MLP",
            tmlp_gdev,
            tmlp_xdev,
            tmlp_gdev / tmlp_xdev,
            tmlp_rev_gdev,
            tmlp_rev_xdev,
            tmlp_rev_gdev / tmlp_rev_xdev,
        ],
    )
    push!(
        df,
        [
            "KAN1",
            tkan1_gdev,
            tkan1_xdev,
            tkan1_gdev / tkan1_xdev,
            tkan1_rev_gdev,
            tkan1_rev_xdev,
            tkan1_rev_gdev / tkan1_rev_xdev,
        ],
    )
    push!(
        df,
        [
            "KAN2",
            tkan2_gdev,
            tkan2_xdev,
            tkan2_gdev / tkan2_xdev,
            tkan2_rev_gdev,
            tkan2_rev_xdev,
            tkan2_rev_gdev / tkan2_rev_xdev,
        ],
    )

    display(df)
    CSV.write(filename, df)

    @info "Results saved to $filename"
    return nothing
end
