using Zygote, Reactant, Enzyme, Test

sumabs2(x) = sum(abs2, x)

@testset "Zygote" begin
    @testset "Zygote.gradient" begin
        x = Reactant.to_rarray(rand(Float32, 32, 10))

        zyg_grad = @jit Zygote.gradient(sumabs2, x)
        enz_grad = @jit Enzyme.gradient(Reverse, Const(sumabs2), x)
        @test zyg_grad[1] isa Reactant.ConcreteRArray
        @test enz_grad[1] ≈ zyg_grad[1]
    end
end
