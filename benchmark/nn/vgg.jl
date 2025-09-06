function setup_vgg_benchmark!(suite::BenchmarkGroup, backend)
    for depth in (16,), bsize in (32,)
        benchmark_name = "VGG$(depth) bn=true (224 x 224 x 3 x $(bsize))"

        setup_lux_benchmark!(
            suite,
            benchmark_name,
            backend,
            Vision.VGG(depth; pretrained=false, batchnorm=true),
            (224, 224, 3, bsize),
        )
    end
end
