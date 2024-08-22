# This will prompt if neccessary to install everything, including CUDA:

using Reactant
using Flux
using Test
# Generate some data for the XOR problem: vectors of length 2, as columns of a matrix:
noisy = rand(Float32, 2, 1000)                                    # 2×1000 Matrix{Float32}
truth = [xor(col[1] > 0.5, col[2] > 0.5) for col in eachcol(noisy)]   # 1000-element Vector{Bool}

# Define our model, a multi-layer perceptron with one hidden layer of size 3:
model = Chain(
    Dense(2 => 3, tanh),   # activation function inside layer
    BatchNorm(3),
    Dense(3 => 2),
    softmax,
)

using BenchmarkTools

origout = model(noisy)

cmodel = Reactant.to_rarray(model)
cnoisy = Reactant.ConcreteRArray(noisy)

# c_o = cmodel(noisy)
# @show c_o[3]
# @btime cmodel(noisy)
# 
# o_c = model(cnoisy)
# @show o_c[3]
# @btime model(cnoisy)
# 
# c_c = cmodel(cnoisy)
# @show c_c[3]
# @btime cmodel(cnoisy)
f = Reactant.compile((a, b) -> a(b), (cmodel, cnoisy))

# using InteractiveUtils
# @show @code_typed f(cmodel,cnoisy)
# @show @code_llvm f(cmodel,cnoisy)
comp = f(cmodel, cnoisy)
# @btime f(cmodel, cnoisy)
@test origout ≈ comp

@testset "conv" begin
    conv = Conv(randn(Float32, 10, 10, 3, 1), randn(Float32, 1))
    conv_reactant = Conv(
        Reactant.ConcreteRArray(conv.weight), Reactant.ConcreteRArray(conv.bias)
    )

    img = randn(Float32, 224, 224, 3, 2)
    img_reactant = Reactant.ConcreteRArray(img)

    comp_conv = Reactant.compile(conv_reactant, (img_reactant,))

    res_reactant = Array{Float32,4}(comp_conv(img_reactant))
    res = conv(img)

    @test res_reactant ≈ res
end

@testset "$f" for f in (NNlib.meanpool, NNlib.maxpool)
    img = randn(Float32, 224, 224, 3, 2)
    img_reactant = Reactant.ConcreteRArray(img)

    f_reactant = Reactant.compile(f, (img_reactant, (3, 3)))

    res_reactant = f_reactant(img_reactant, (3, 3))
    res = f(img, (3, 3))

    @test res_reactant ≈ res
end
