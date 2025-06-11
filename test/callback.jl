using Reactant

function fn(x)
    error("This should error at runtime")
    return x .+ 1
end

@testset "error" begin
    x = Reactant.to_rarray(ones(4))

    hlo = repr(@code_hlo fn(x))
    @test contains(hlo, "stablehlo.custom_call")

    fn_compiled = @compile fn(x)
    @test_throws Reactant.XLA.ReactantInternalError fn_compiled(x)
end
