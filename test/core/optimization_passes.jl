using Reactant, Test

# The `:all` pipeline finishes with a cleanup stage that re-runs the pattern set with
# transpose/reshape propagation reversed. The concat/slice-to-batch rewrites have to stay
# out of that stage: with propagation reversed, `concat_insert_dim_elementwise` and the
# reshape-propagation patterns rewrite each other in a cycle and the greedy driver never
# reaches a fixed point (layered bf16 graphs hung `@compile` indefinitely). Batching has
# already been applied by the main pipeline at that point.

function cleanup_pipelines(compile_options)
    cleanup = Reactant.__compile_options_without_batching_passes(compile_options)
    down = Reactant.Compiler.optimization_passes(
        Reactant.__compile_options_with_reversed_propagation(cleanup);
        backend="cpu",
        raise_shlo_to_blas_lapack=false,
    )
    up = Reactant.Compiler.optimization_passes(
        cleanup; backend="cpu", raise_shlo_to_blas_lapack=false
    )
    return down, up
end

@testset "post-optimization cleanup" begin
    compile_options = Reactant.CompileOptions()
    down, up = cleanup_pipelines(compile_options)

    @testset "excludes the batching rewrites" begin
        for pipeline in (down, up)
            @test !contains(pipeline, "concat_insert_dim")
            @test !contains(pipeline, "slice_to_batch")
            @test contains(pipeline, "transform-interpreter")
        end
        # ... while the main pipeline still runs them
        @test contains(
            Reactant.Compiler.optimization_passes(compile_options; backend="cpu"),
            "concat_insert_dim",
        )
    end

    @testset "terminates on convert(concat(reshape))" begin
        # Minimal trigger of the cycle: `convert_concat` pushes the convert into the
        # operands, `elementwise_reshape_like` hoists the reshapes back out, and
        # `concat_insert_dim_elementwise` batches the converts back into this module.
        # Without the exclusion this call never returns.
        source = """
        module {
          func.func @main(%a: tensor<16x64x2xf32>, %b: tensor<16x64x2xf32>) -> tensor<2x16x64x2xbf16> {
            %ra = stablehlo.reshape %a : (tensor<16x64x2xf32>) -> tensor<1x16x64x2xf32>
            %rb = stablehlo.reshape %b : (tensor<16x64x2xf32>) -> tensor<1x16x64x2xf32>
            %c = stablehlo.concatenate %ra, %rb, dim = 0 : (tensor<1x16x64x2xf32>, tensor<1x16x64x2xf32>) -> tensor<2x16x64x2xf32>
            %cv = stablehlo.convert %c : (tensor<2x16x64x2xf32>) -> tensor<2x16x64x2xbf16>
            return %cv : tensor<2x16x64x2xbf16>
          }
        }
        """
        out = repr(
            Reactant.Compiler.run_pass_pipeline_on_source(
                source, join([down, up, down], ",")
            ),
        )
        @test contains(out, "stablehlo.concatenate")
        @test contains(out, "stablehlo.convert")
        @test contains(out, "tensor<2x16x64x2xbf16>")
    end
end
