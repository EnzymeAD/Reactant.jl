using Reactant, Lux, Random, Statistics
using Enzyme
using Test

# Generate some data for the XOR problem: vectors of length 2, as columns of a matrix:
noisy = rand(Float32, 2, 1000)                                        # 2×1000 Matrix{Float32}
truth = [xor(col[1] > 0.5, col[2] > 0.5) for col in eachcol(noisy)]   # 1000-element Vector{Bool}

# Define our model, a multi-layer perceptron with one hidden layer of size 3:
model = Lux.Chain(
    Lux.Dense(2 => 3, tanh),   # activation function inside layer
    Lux.Dense(3 => 2),
    softmax,
)
ps, st = Lux.setup(Xoshiro(123), model)

using BenchmarkTools

origout, _ = model(noisy, ps, st)
@show origout[3]
@btime model($noisy, $ps, $st)  # 52.731 μs (10 allocations: 32.03 KiB)

cmodel = Reactant.make_tracer(IdDict(), model, (), Reactant.ArrayToConcrete)
cps = Reactant.make_tracer(IdDict(), ps, (), Reactant.ArrayToConcrete)
cst = Reactant.make_tracer(IdDict(), st, (), Reactant.ArrayToConcrete)
cnoisy = Reactant.ConcreteRArray(noisy)

f = Reactant.compile((a, b, c, d) -> first(a(b, c, d)), (cmodel, cnoisy, cps, cst))

# # using InteractiveUtils
# # @show @code_typed f(cmodel,cnoisy)
# # @show @code_llvm f(cmodel,cnoisy)
comp = f(cmodel, cnoisy, cps, cst)
@show comp[3]
@btime f($cmodel, $cnoisy, $cps, $cst) # 4.430 μs (5 allocations: 160 bytes)

# To train the model, we use batches of 64 samples, and one-hot encoding:

using MLUtils, OneHotArrays, Optimisers

target = onehotbatch(truth, [true, false])                   # 2×1000 OneHotMatrix
ctarget = Reactant.ConcreteRArray(Array{Float32}(target))
loader = DataLoader((noisy, target); batchsize=64, shuffle=true);
# # 16-element DataLoader with first element: (2×64 Matrix{Float32}, 2×64 OneHotMatrix)

opt = Optimisers.Adam(0.01f0)
losses = []

# Lux.Exprimental.TrainState is very specialized for Lux models, so we write out the
# training loop manually:
function crossentropy(ŷ, y)
    logŷ = log.(ŷ)
    result = y .* logŷ
    # result = ifelse.(y .== 0.0f0, zero.(result), result)
    return -sum(result)
end

function loss_function(model, x, y, ps, st)
    y_hat, _ = model(x, ps, st)
    return crossentropy(y_hat, y)
end

function gradient_loss_function(model, x, y, ps, st)
    dps = Enzyme.make_zero(ps)
    _, res = Reactant.autodiff(
        ReverseWithPrimal,
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

gradient_loss_function(model, noisy, target, ps, st)

compiled_gradient = Reactant.compile(
    gradient_loss_function, (cmodel, cnoisy, ctarget, cps, cst)
)

# # Training loop, using the whole data set 1000 times:
# losses = []
# for epoch in 1:1_000
#     for (x, y) in loader
#         loss, grads = Flux.withgradient(model) do m
#             # Evaluate model and loss inside gradient context:
#             y_hat = m(x)
#             return Flux.crossentropy(y_hat, y)
#         end
#         Flux.update!(optim, model, grads[1])
#         push!(losses, loss)  # logging, outside gradient context
#     end
# end

# optim # parameters, momenta and output have all changed
# out2 = model(noisy)  # first row is prob. of true, second row p(false)

# mean((out2[1, :] .> 0.5) .== truth)  # accuracy 94% so far!
