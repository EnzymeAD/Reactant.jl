# Neural Network Benchmarks
module NN

using Boltz: Vision
using Lux: Lux
using Reactant: Reactant, @compile
using Enzyme: Enzyme

using BenchmarkTools: BenchmarkGroup, @benchmarkable
using Random: Random

include("nn/common.jl")
include("nn/vit.jl")
include("nn/vgg.jl")

end

function setup_benchmarks!(suite::BenchmarkGroup, backend::String)
    # neural network benchmarks
    ## vision models
    NN.setup_vit_benchmark!(suite, backend)
    NN.setup_vgg_benchmark!(suite, backend)

    ## neural operator benchmarks

    return nothing
end
