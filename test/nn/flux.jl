using Reactant, Flux

@testset "Flux.jl Integration" begin
    # Generate some data for the XOR problem: vectors of length 2, as columns of a matrix:
    noisy = rand(Float32, 2, 1000)                                        # 2×1000 Matrix{Float32}
    truth = [xor(col[1] > 0.5, col[2] > 0.5) for col in eachcol(noisy)]   # 1000-element Vector{Bool}

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
