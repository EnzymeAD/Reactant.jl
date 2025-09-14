function run_vit_benchmark!(results, backend)
    lowercase(backend) == "cpu" && return nothing
    for mode in (:tiny,), bsize in (4,)
        benchmark_name = "ViT $(mode) [256, 256, 3, $(bsize)]"

        # backward benchmarks for this are extremely slow hence the reduced set
        run_lux_benchmark!(
            results,
            benchmark_name,
            backend,
            Vision.ViT(mode),
            (256, 256, 3, bsize);
            bwd_enzyme_pass_options=(:all,),
            disable_bwd_scatter_gather_bench=false,
            disable_bwd_pad_bench=false,
        )
    end
end

function run_vgg_benchmark!(results, backend)
    lowercase(backend) == "cpu" && return nothing
    for depth in (11,), bsize in (4,)
        benchmark_name = "VGG$(depth) bn=true [224, 224, 3, $(bsize)]"

        run_lux_benchmark!(
            results,
            benchmark_name,
            backend,
            Vision.VGG(depth; pretrained=false, batchnorm=true),
            (224, 224, 3, bsize),
        )
    end
end
