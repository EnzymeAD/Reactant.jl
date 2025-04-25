using Reactant, Test

f1(x::AbstractMatrix) = sum(x; dims=1)

@testset "mapslices" begin
    A = collect(reshape(1:30, (2, 5, 3)))
    A_ra = Reactant.to_rarray(A)

    B = mapslices(f1, A; dims=[1, 2])
    B_ra = @jit mapslices(f1, A_ra; dims=[1, 2])

    @test B ≈ B_ra

    B = mapslices(sum, A; dims=[1, 3])
    B_ra = @jit mapslices(sum, A_ra; dims=[1, 3])

    @test B ≈ B_ra
end
