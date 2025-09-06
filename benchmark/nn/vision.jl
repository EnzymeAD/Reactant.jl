function setup_vit_benchmark!(suite::BenchmarkGroup, backend)
    for mode in (:small,), bsize in (4,)
        benchmark_name = "ViT $(mode) (256 x 256 x 3 x $(bsize))"

        setup_lux_benchmark!(
            suite, benchmark_name, backend, Vision.ViT(mode), (256, 256, 3, bsize)
        )
    end
end

function setup_vgg_benchmark!(suite::BenchmarkGroup, backend)
    for depth in (16,), bsize in (4,)
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
