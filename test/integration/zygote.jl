using Zygote, Reactant, Enzyme, Test

sumabs2(x) = sum(abs2, x)

@testset "Zygote" begin
    @testset "Zygote.gradient" begin
        x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 32, 10))

        zyg_grad = @jit Zygote.gradient(sumabs2, x)
        enz_grad = @jit Enzyme.gradient(Reverse, Const(sumabs2), x)
        @test zyg_grad[1] isa Reactant.ConcreteRArray
        @test enz_grad[1] ≈ zyg_grad[1]

        @testset "Disable Overlay" begin
            @test_throws Zygote.CompileError Reactant.with_config(;
                overlay_zygote_calls=false
            ) do
                @jit Zygote.gradient(sumabs2, x)
            end
        end
    end
end
