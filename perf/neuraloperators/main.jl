using NeuralOperators, Lux, Random

include("../common.jl")

const xdev = reactant_device()

function run_deeponet_benchmarks()
    @info "Running DeepONet benchmarks"

    model = DeepONet(;
        branch=(64, ntuple(Returns(256), 5)..., 16),
        trunk=(1, ntuple(Returns(256), 5)..., 16),
        branch_activation=gelu,
        trunk_activation=gelu,
    )
    ps, st = xdev(Lux.setup(Random.default_rng(), model))
    u = xdev(rand(Float32, 64, 1024))
    y = xdev(rand(Float32, 1, 128))

    primal_timings = Reactant.with_config(;
        dot_general_precision=PrecisionConfig.HIGH,
        convolution_precision=PrecisionConfig.HIGH,
    ) do
        benchmark_nn_primal(
            model,
            (u, y),
            ps,
            st;
            disable_scatter_gather_bench=true,
            disable_pad_bench=true,
        )
    end

    pretty_print_table(permutedims(hcat([[t...] for t in primal_timings]...), (2, 1)))

    return nothing
end

run_deeponet_benchmarks()
