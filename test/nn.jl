# This will prompt if neccessary to install everything, including CUDA:

using Reactant
using Flux
using Test
# Generate some data for the XOR problem: vectors of length 2, as columns of a matrix:
noisy = rand(Float32, 2, 1000)                                    # 2×1000 Matrix{Float32}
truth = [xor(col[1]>0.5, col[2]>0.5) for col in eachcol(noisy)]   # 1000-element Vector{Bool}

# Define our model, a multi-layer perceptron with one hidden layer of size 3:
model = Chain(
    Dense(2 => 3, tanh),   # activation function inside layer
    BatchNorm(3),
    Dense(3 => 2),
    softmax
)

using BenchmarkTools

origout = model(noisy)
@show origout[3]
@btime model(noisy)

cmodel = Reactant.make_tracer(IdDict(), model, (), Reactant.ArrayToConcrete, nothing)
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
f=Reactant.compile((a,b)->a(b), (cmodel,cnoisy))

# using InteractiveUtils
# @show @code_typed f(cmodel,cnoisy)
# @show @code_llvm f(cmodel,cnoisy)
comp = f(cmodel,cnoisy)
@show comp[3]
@btime f(cmodel,cnoisy)

# To train the model, we use batches of 64 samples, and one-hot encoding:
target = Flux.onehotbatch(truth, [true, false])                   # 2×1000 OneHotMatrix
loader = Flux.DataLoader((noisy, target), batchsize=64, shuffle=true);
# 16-element DataLoader with first element: (2×64 Matrix{Float32}, 2×64 OneHotMatrix)

optim = Flux.setup(Flux.Adam(0.01), model)  # will store optimiser momentum, etc.

# Training loop, using the whole data set 1000 times:
losses = []
for epoch in 1:1_000
    for (x, y) in loader
        loss, grads = Flux.withgradient(model) do m
            # Evaluate model and loss inside gradient context:
            y_hat = m(x)
            Flux.crossentropy(y_hat, y)
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, loss)  # logging, outside gradient context
    end
end

optim # parameters, momenta and output have all changed
out2 = model(noisy)  # first row is prob. of true, second row p(false)

mean((out2[1,:] .> 0.5) .== truth)  # accuracy 94% so far!
