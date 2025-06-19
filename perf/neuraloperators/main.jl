using NeuralOperators, Lux, Random

include("../common.jl")

const xdev = reactant_device(; force=true)

function run_deeponet_benchmarks()
    @info "Running DeepONet benchmarks"

    model = DeepONet(;
        branch=(64, ntuple(Returns(256), 4)..., 16),
        trunk=(1, ntuple(Returns(256), 4)..., 16),
        branch_activation=gelu,
        trunk_activation=gelu,
    )
    ps, st = xdev(Lux.setup(Random.default_rng(), model))
    u = xdev(rand(Float32, 64, 1024))
    y = xdev(rand(Float32, 1, 128))
    z = xdev(rand(Float32, 128, 1024))

    primal_timings = Reactant.with_config(;
        dot_general_precision=PrecisionConfig.HIGH,
        convolution_precision=PrecisionConfig.HIGH,
    ) do
        benchmark_nn_primal(
            model,
            (u, y),
            z,
            ps,
            st;
            disable_scatter_gather_bench=true,
            disable_pad_bench=true,
        )
    end

    gradient_timings = Reactant.with_config(;
        dot_general_precision=PrecisionConfig.HIGH,
        convolution_precision=PrecisionConfig.HIGH,
    ) do
        benchmark_nn_gradient(
            model,
            (u, y),
            z,
            ps,
            st;
            disable_scatter_gather_bench=true,
            disable_pad_bench=true,
        )
    end

    timings = vcat(primal_timings, gradient_timings)
    pretty_print_table(permutedims(hcat([[t...] for t in timings]...), (2, 1)))

    return nothing
end

function run_fno_benchmarks()
    @info "Running FNO benchmarks"

    model = FourierNeuralOperator((16, 16), 3, 8, 64)
    ps, st = xdev(Lux.setup(Random.default_rng(), model))
    x = xdev(rand(Float32, 64, 64, 1, 256))
    z = xdev(rand(Float32, 64, 64, 8, 256))

    primal_timings = Reactant.with_config(;
        dot_general_precision=PrecisionConfig.HIGH,
        convolution_precision=PrecisionConfig.HIGH,
    ) do
        benchmark_nn_primal(
            model,
            x,
            z,
            ps,
            st;
            disable_scatter_gather_bench=true,
            disable_pad_bench=true,
        )
    end

    gradient_timings = Reactant.with_config(;
        dot_general_precision=PrecisionConfig.HIGH,
        convolution_precision=PrecisionConfig.HIGH,
    ) do
        benchmark_nn_gradient(
            model,
            x,
            z,
            ps,
            st;
            disable_scatter_gather_bench=true,
            disable_pad_bench=true,
        )
    end

    timings = vcat(primal_timings, gradient_timings)
    pretty_print_table(permutedims(hcat([[t...] for t in timings]...), (2, 1)))

    return nothing
end

run_deeponet_benchmarks()
run_fno_benchmarks()
