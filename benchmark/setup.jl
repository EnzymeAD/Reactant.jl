# Neural Network Benchmarks
module NN

using Boltz: Vision
using Lux: Lux, gelu, reactant_device
using Printf: @sprintf
using Reactant: Reactant, @compile
using NeuralOperators: DeepONet, FourierNeuralOperator
using Enzyme: Enzyme

using Chairmarks: @b
using Random: Random

include("nn/common.jl")
include("nn/vision.jl")
include("nn/neural_operators.jl")

end

function run_benchmarks(backend::String)
    results = Dict()

    # neural network benchmarks
    ## vision models
    NN.run_vgg_benchmark!(results, backend)
    NN.run_vit_benchmark!(results, backend)

    ## neural operator benchmarks
    NN.run_deeponet_benchmark!(results, backend)
    NN.run_fno_benchmark!(results, backend)

    return results
end
