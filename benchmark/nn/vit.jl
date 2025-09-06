function setup_vit_benchmark!(suite::BenchmarkGroup, backend)
    for mode in (:tiny, :small), bsize in (32,)
        benchmark_name = "ViT $(mode) (256 x 256 x 3 x $(bsize))"

        setup_lux_benchmark!(
            suite, benchmark_name, backend, Vision.ViT(mode), (256, 256, 3, bsize)
        )
    end
end
