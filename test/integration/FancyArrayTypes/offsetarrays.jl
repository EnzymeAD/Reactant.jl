using Reactant, Test, OffsetArrays

function scalar_index(x)
    @allowscalar getindex(x, -1, 0)
end

@testset "OffsetArrays" begin
    A = Float64.(reshape(1:15, 3, 5))
    OA = OffsetArray(A, -1:1, 0:4)
    rOA = Reactant.to_rarray(OA)

    oval = scalar_index(OA)
    cval = scalar_index(rOA)
    @test cval ≈ oval

    tval = @jit scalar_index(rOA)
    @test tval ≈ oval
end
