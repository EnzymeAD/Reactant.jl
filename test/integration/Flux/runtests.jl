using Reactant, Flux, Test

@testset "Flux.jl Integration" begin
    noisy = rand(Float32, 2, 1000)
    truth = [xor(col[1] > 0.5, col[2] > 0.5) for col in eachcol(noisy)]

    model = Chain(
        Dense(2 => 3, tanh),
        BatchNorm(3),
        Dense(3 => 2),
        softmax,
    )

    origout = model(noisy)

    cmodel = Reactant.to_rarray(model)
    cnoisy = Reactant.to_rarray(noisy)

    fn = (a, b) -> a(b)
    f = @compile fn(cmodel, cnoisy)

    comp = f(cmodel, cnoisy)
    @test origout ≈ comp atol = 1e-3 rtol = 1e-2
end
