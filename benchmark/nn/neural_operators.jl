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
    lowercase(backend) == "cpu" && return nothing
    model = FourierNeuralOperator((16, 16), 3, 8, 64)

    run_lux_benchmark!(results, "FNO [64, 64, 1, 4]", backend, model, (64, 64, 1, 4))

    return nothing
end
