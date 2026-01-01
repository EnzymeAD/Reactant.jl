using Boltz: Vision

include("common.jl")

function run_vit_benchmark!(results, backend)
    lowercase(backend) == "cpu" && return nothing
    for mode in (:tiny,), bsize in (4,)
        # backward benchmarks for this are extremely slow hence the reduced set
        run_lux_benchmark!(
            results,
            "ViT $(mode) [256, 256, 3, $(bsize)]",
            backend,
            Vision.ViT(mode),
            (256, 256, 3, bsize);
            bwd_enzyme_pass_options=(:all,),
            disable_bwd_scatter_gather_bench=false,
            disable_bwd_pad_bench=false,
            disable_bwd_transpose_bench=false,
        )
    end
end

function run_vgg_benchmark!(results, backend)
    lowercase(backend) == "cpu" && return nothing
    for depth in (11,), bsize in (4,)
        run_lux_benchmark!(
            results,
            "VGG$(depth) bn=true [224, 224, 3, $(bsize)]",
            backend,
            Vision.VGG(depth; pretrained=false, batchnorm=true),
            (224, 224, 3, bsize),
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    backend = get_backend()
    results = Dict()
    run_vit_benchmark!(results, backend)
    run_vgg_benchmark!(results, backend)
end
