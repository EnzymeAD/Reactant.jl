using Reactant, Lux, Random, Statistics, Enzyme, Functors, OneHotArrays

function loss_function(model, x, y, ps, st)
    y_hat, _ = model(x, ps, st)
    return CrossEntropyLoss()(y_hat, y)
end

function gradient_loss_function(model, x, y, ps, st)
    dps = Enzyme.make_zero(ps)
    _, res = Enzyme.autodiff(
        set_runtime_activity(ReverseWithPrimal),
        loss_function,
        Active,
        Const(model),
        Const(x),
        Const(y),
        Duplicated(ps, dps),
        Const(st),
    )
    return res, dps
end

@testset "Lux.jl Integration" begin
    # Generate some data for the XOR problem: vectors of length 2, as columns of a matrix:
    noisy = rand(Float32, 2, 1000)                                        # 2×1000 Matrix{Float32}
    truth = [xor(col[1] > 0.5, col[2] > 0.5) for col in eachcol(noisy)]   # 1000-element Vector{Bool}

    # Define our model, a multi-layer perceptron with one hidden layer of size 3:
    model = Lux.Chain(
        Lux.Dense(2 => 3, tanh),   # activation function inside layer
        Lux.BatchNorm(3, sigmoid),
        Lux.Dense(3 => 2),
        softmax,
    )
    ps, st = Lux.setup(Xoshiro(123), model)

    origout, _ = model(noisy, ps, Lux.testmode(st))

    cmodel = Reactant.to_rarray(model)
    cps = Reactant.to_rarray(ps)
    cst = Reactant.to_rarray(Lux.testmode(st))
    cst2 = Reactant.to_rarray(st)
    cnoisy = Reactant.ConcreteRArray(noisy)

    @info @__LINE__
    f = Reactant.compile((a, b, c, d) -> first(a(b, c, d)), (cmodel, cnoisy, cps, cst))
    @info @__LINE__
    f_mlir = Reactant.Compiler.compile_mlir((a, b, c, d) -> first(a(b, c, d)), (cmodel, cnoisy, cps, cst))[1]
    @info @__LINE__
    println(f_mlir)
    @info @__LINE__

    comp = f(cmodel, cnoisy, cps, cst)
    @info @__LINE__

    @test comp ≈ origout atol = 1e-3 rtol = 1e-2

    target = onehotbatch(truth, [true, false])                   # 2×1000 OneHotMatrix
    @info @__LINE__

    ctarget = Reactant.ConcreteRArray(Array{Float32}(target))
    # ctarget = Reactant.to_rarray(target)
    @info @__LINE__

    res, dps = gradient_loss_function(model, noisy, target, ps, st)
    @info @__LINE__

    compiled_gradient = Reactant.compile(
        gradient_loss_function, (cmodel, cnoisy, ctarget, cps, cst2)
    )
    @info @__LINE__
    compiled_gradient_mlir = Reactant.Compiler.compile_mlir(
        gradient_loss_function, (cmodel, cnoisy, ctarget, cps, cst2)
    )[1]
    @info @__LINE__

    println(compiled_gradient_mlir)
    @info @__LINE__

    res_reactant, dps_reactant = compiled_gradient(cmodel, cnoisy, ctarget, cps, cst2)
    @info @__LINE__
    res_reactant_mlir, dps_reactant_mlir = @jit Reactant.Ops.hlo_call(repr(compiled_gradient_mlir), cmodel, cnoisy, ctarget, cps, cst2)
    @info @__LINE__

    @test res ≈ res_reactant_mlir atol = 1e-3 rtol = 1e-2
    for (dps1, dps2) in zip(fleaves(dps), fleaves(dps_reactant_mlir))
        @test dps1 ≈ dps2 atol = 1e-3 rtol = 1e-2
    end
end
