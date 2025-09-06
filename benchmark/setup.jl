# Neural Network Benchmarks
module NN

using Boltz: Vision
using Lux: Lux, gelu
using Reactant: Reactant, @compile
using NeuralOperators: DeepONet, FourierNeuralOperator
using Enzyme: Enzyme

using BenchmarkTools: BenchmarkGroup, @benchmarkable
using Random: Random

include("nn/common.jl")
include("nn/vision.jl")
include("nn/neural_operators.jl")

end

function setup_benchmarks!(suite::BenchmarkGroup, backend::String)
    # neural network benchmarks
    ## vision models
    NN.setup_vgg_benchmark!(suite, backend)
    NN.setup_vit_benchmark!(suite, backend)

    ## neural operator benchmarks
    NN.setup_deeponet_benchmark!(suite, backend)
    # XXX: some pass broke this
    # NN.setup_fno_benchmark!(suite, backend)

    return nothing
end
