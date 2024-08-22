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

cmodel = Reactant.make_tracer(IdDict(), model, (), Reactant.ArrayToConcrete)
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
