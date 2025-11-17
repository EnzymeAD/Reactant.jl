using Reactant, Enzyme, Lux, Random, Test

original_backend = Reactant.XLA.default_backend()
Reactant.set_default_backend("metal")

sincos(x) = sin(cos(x))
sumsincos(x) = sum(sincos, x)

@testset "Simple Function" begin
    x = reshape(collect(Float32, 1:40), 10, 4)
    x_ra = Reactant.to_rarray(x)

    @test @jit(sincos(x_ra)) ≈ sincos(x)
end

@testset "Autodiff" begin
    x = reshape(collect(Float32, 1:40), 10, 4)
    x_ra = Reactant.to_rarray(x)

    @test @jit(sumsincos(x_ra)) ≈ sum(sincos, x)

    @test @jit(Enzyme.gradient(Enzyme.Reverse, sumsincos, x_ra))[2] ≈
        Enzyme.gradient(Enzyme.Reverse, sumsincos, x)[2]
    @test @jit(Enzyme.gradient(Enzyme.Forward, sumsincos, x_ra))[2] ≈
        Enzyme.gradient(Enzyme.Forward, sumsincos, x)[2]
end

@testset "CNN" begin
    model = Chain(
        Conv((5, 5), 1 => 6, relu),
        MaxPool((2, 2)),
        Conv((5, 5), 6 => 16, relu),
        MaxPool((2, 2)),
        FlattenLayer(3),
        Chain(Dense(256 => 128, relu), Dense(128 => 84, relu), Dense(84 => 10)),
    )
    ps, st = Lux.setup(Random.default_rng(), model)
    x = Reactant.TestUtils.construct_test_array(Float32, 28, 28, 1, 4)

    st_test = Lux.testmode(st)

    ps_ra = Reactant.to_rarray(ps)
    st_ra = Reactant.to_rarray(st)
    x_ra = Reactant.to_rarray(x)

    st_ra_test = Lux.testmode(st_ra)

    @test @jit(model(x_ra, ps_ra, st_ra_test))[1] ≈ model(x, ps, st_test)[1]
end

Reactant.set_default_backend(original_backend)
