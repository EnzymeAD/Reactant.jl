function run_vit_benchmark!(results, backend)
    for mode in (:tiny,), bsize in (4,)
        benchmark_name = "ViT $(mode) [256, 256, 3, $(bsize)]"

        run_lux_benchmark!(
            results, benchmark_name, backend, Vision.ViT(mode), (256, 256, 3, bsize)
        )
    end
end

function run_vgg_benchmark!(results, backend)
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
