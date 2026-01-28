using NeuralOperators: DeepONet, FourierNeuralOperator
using Lux: gelu

include("common.jl")

function run_deeponet_benchmark!(results, backend)
    model = DeepONet(;
        branch=(64, ntuple(Returns(256), 4)..., 16),
        trunk=(1, ntuple(Returns(256), 4)..., 16),
        branch_activation=gelu,
        trunk_activation=gelu,
    )

    run_lux_benchmark!(
        results, "DeepONet ([64, 1024], [1, 128])", backend, model, ((64, 1024), (1, 128))
    )

    return nothing
end

function run_fno_benchmark!(results, backend)
    model = FourierNeuralOperator((16, 16), 3, 8, 64)

    run_lux_benchmark!(results, "FNO [64, 64, 1, 4]", backend, model, (64, 64, 1, 4))

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    backend = get_backend()
    results = Dict()
    run_deeponet_benchmark!(results, backend)
    run_fno_benchmark!(results, backend)
end
