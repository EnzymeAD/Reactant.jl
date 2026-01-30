using Reactant, Test, Statistics

@testset "Statistics: `mean` & `var`" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 3, 4)
    x_ca = Reactant.to_rarray(x)

    @test @jit(mean(x_ca)) ≈ mean(x)
    @test @jit(mean(x_ca; dims=1)) ≈ mean(x; dims=1)
    @test @jit(mean(x_ca; dims=(1, 2))) ≈ mean(x; dims=(1, 2))
    @test @jit(mean(x_ca; dims=(1, 3))) ≈ mean(x; dims=(1, 3))

    @test @jit(var(x_ca)) ≈ var(x)
    @test @jit(var(x_ca, dims=1)) ≈ var(x; dims=1)
    @test @jit(var(x_ca, dims=(1, 2); corrected=false)) ≈
        var(x; dims=(1, 2), corrected=false)
    @test @jit(var(x_ca; dims=(1, 3), corrected=false)) ≈
        var(x; dims=(1, 3), corrected=false)
end
