using Reactant, Lux, Random, Statistics, Enzyme, Functors, OneHotArrays

function loss_function(model, x, y, ps, st)
    y_hat, _ = model(x, ps, st)
    return CrossEntropyLoss()(y_hat, y)
end

function loss_function(model, x, ps, st)
    y_hat, _ = model(x, ps, st)
    return sum(abs2, y_hat)
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

function gradient_loss_function(model, x, ps, st)
    dps = Enzyme.make_zero(ps)
    _, res = Enzyme.autodiff(
        set_runtime_activity(ReverseWithPrimal),
        loss_function,
        Active,
        Const(model),
        Const(x),
        Duplicated(ps, dps),
        Const(st),
    )
    return res, dps
end

@testset "Lux.jl Integration" begin
    # Generate some data for the XOR problem: vectors of length 2, as columns of a matrix:
    noisy = Reactant.TestUtils.construct_test_array(Float32, 2, 1000)
    truth = Reactant.TestUtils.construct_test_array(Int, 1000) .> 500

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
    cnoisy = Reactant.to_rarray(noisy)

    f = Reactant.compile((a, b, c, d) -> first(a(b, c, d)), (cmodel, cnoisy, cps, cst))

    comp = f(cmodel, cnoisy, cps, cst)

    @test comp ≈ origout atol = 1e-3 rtol = 1e-2

    target = onehotbatch(truth, [true, false])
    ctarget = Reactant.to_rarray(target)

    # Gradient correctness tests are run in luxlib.jl and nnlib.jl
    res_reactant, dps_reactant = @jit gradient_loss_function(
        cmodel, cnoisy, ctarget, cps, cst2
    )
    @test res_reactant isa Reactant.ConcreteRNumber
end

@testset "RNN Integration" begin
    model = Recurrence(RNNCell(4 => 4); ordering=BatchLastIndex())
    ps, st = Reactant.to_rarray(Lux.setup(Random.default_rng(), model))

    x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 4, 16, 12))

    # This test requires running optimizations between the enzyme autodiff passes
    res, ∂ps = @jit gradient_loss_function(model, x, ps, st)
    @test res isa Reactant.ConcreteRNumber
end

@testset "RNG stored in state" begin
    model = Dropout(0.5f0)
    ps, st = Reactant.to_rarray(Lux.setup(Random.default_rng(), model))

    x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 10, 10))

    res, st_new = @jit model(x, ps, st)
    @test st_new.rng isa Reactant.ReactantRNG
end
