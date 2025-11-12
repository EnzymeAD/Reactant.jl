using Reactant, Flux, Test

@testset "Flux.jl Integration" begin
    # Generate some data for the XOR problem: vectors of length 2, as columns of a matrix:
    noisy = Reactant.TestUtils.construct_test_array(Float32, 2, 1000)
    truth = Reactant.TestUtils.construct_test_array(Int, 1000) .> 500

    # Define our model, a multi-layer perceptron with one hidden layer of size 3:
    model = Chain(
        Dense(2 => 3, tanh),   # activation function inside layer
        BatchNorm(3),
        Dense(3 => 2),
        softmax,
    )

    origout = model(noisy)

    cmodel = Reactant.to_rarray(model)
    cnoisy = Reactant.to_rarray(noisy)

    f = Reactant.compile((a, b) -> a(b), (cmodel, cnoisy))

    comp = f(cmodel, cnoisy)
    @test origout ≈ comp atol = 1e-3 rtol = 1e-2
end
