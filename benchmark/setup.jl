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
include("nn/dgcnn.jl")

end

# Miscellaneous Benchmarks
module Misc

using Reactant, LinearAlgebra
using Chairmarks: @b
using Printf: @sprintf
using Random: Random

include("misc/common.jl")
include("misc/newton_schulz.jl")

end

module Polybench

using Reactant, LinearAlgebra
using Chairmarks: @b
using Printf: @sprintf
using Random: Random

include("polybench/common.jl")
include("polybench/stencil.jl")
include("polybench/data_mining.jl")
include("polybench/blas.jl")
include("polybench/linalg_kernels.jl")

end

function run_benchmarks(backend::String)
    results = Dict()

    # polybench benchmarks
    Polybench.run_data_mining_benchmarks!(results, backend)
    Polybench.run_blas_benchmarks!(results, backend)
    Polybench.run_linalg_kernel_benchmarks!(results, backend)
    Polybench.run_stencil_benchmarks!(results, backend)

    # neural network benchmarks
    NN.run_vgg_benchmark!(results, backend)
    NN.run_vit_benchmark!(results, backend)
    NN.run_deeponet_benchmark!(results, backend)
    NN.run_fno_benchmark!(results, backend)
    NN.run_dgcnn_benchmark!(results, backend)

    # misc benchmarks
    Misc.run_newton_schulz_benchmark!(results, backend)

    return results
end
