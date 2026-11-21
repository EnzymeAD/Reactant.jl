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

struct LocationEmbInputWrapper{A}
    add_loc::Bool
    embedding::A
end

Flux.@layer LocationEmbInputWrapper trainable = (embedding,)

function (m::LocationEmbInputWrapper)(xs)
    if m.add_loc
        batch_size = size(xs)[end]
        emb = repeat(m.embedding, 1, 1, 1, batch_size)
        return cat(xs, emb; dims=3)
    else
        return xs
    end
end

@testset "Flux layer show without scalar indexing (#3554)" begin
    dev = reactant_device()
    wrapper = dev(LocationEmbInputWrapper(true, zeros(Float32, 8, 8, 6)))
    chain = dev(Chain(Dense(4 => 3, relu), BatchNorm(3), Dense(3 => 2)))
    w_view = @view(Reactant.to_rarray(ones(Float32, 4, 4))[:, 1:3])
    dense_view = Dense(w_view, Reactant.to_rarray(zeros(Float32, 4)))

    task_local_storage(:ScalarIndexing, Reactant.GPUArraysCore.ScalarDisallowed) do
        @test occursin("all zero", sprint(show, MIME"text/plain"(), wrapper))
        @test occursin("Total:", sprint(show, MIME"text/plain"(), chain))
        @test occursin("parameters", sprint(show, MIME"text/plain"(), dense_view))
    end

    x = Reactant.to_rarray(ones(Float32, 8, 8, 1, 2))
    y = @jit wrapper(x)
    @test size(y) == (8, 8, 7, 2)
end
