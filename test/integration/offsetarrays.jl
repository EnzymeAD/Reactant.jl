using Reactant
using Test
using OffsetArrays

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

@testset "OffsetArray View" begin
    U = zeros(Float64, 128, 128, 1)
    vU = OffsetArray(U, -7:120, -7:120, 1:1)
    rU = Reactant.to_rarray(vU)

    @jit fill!(@view(rU[1:112, 1:112, 1]), 1.0)
    fill!(@view(vU[1:112, 1:112, 1]), 1.0)

    @test parent(rU) ≈ parent(vU)
end
