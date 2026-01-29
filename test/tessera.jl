using Reactant, Test

@tessera_op "reciprocal" function reciprocal(x)
    return 1 ./ x
end


@trace tessera function foo(x)
    return sin.(sum(x) .+ x)
end


@testset "Tessera Annotation Tests" begin
    x = Reactant.to_rarray(rand(3))
    # if optimize=false is not set, the function is inlined.
    hlo = repr(@code_hlo optimize = false reciprocal(x))
    @test occursin("tessera_op = \"reciprocal\"", hlo)

    hlo2 = repr(@code_hlo optimize = false foo(x))
    @test occursin("tessera_op = \"foo\"", hlo2)
end