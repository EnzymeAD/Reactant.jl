function setup_deeponet_benchmark!(suite::BenchmarkGroup, backend)
    model = DeepONet(;
        branch=(64, ntuple(Returns(256), 4)..., 16),
        trunk=(1, ntuple(Returns(256), 4)..., 16),
        branch_activation=gelu,
        trunk_activation=gelu,
    )

    benchmark_name = "DeepONet ((64 x 1024), (1 x 128))"
    setup_lux_benchmark!(suite, benchmark_name, backend, model, ((64, 1024), (1, 128)))

    return nothing
end

function setup_fno_benchmark!(suite::BenchmarkGroup, backend)
    model = FourierNeuralOperator((16, 16), 3, 8, 64)

    benchmark_name = "FNO (64 x 64 x 1 x 256)"
    setup_lux_benchmark!(suite, benchmark_name, backend, model, (64, 64, 1, 256))

    return nothing
end
