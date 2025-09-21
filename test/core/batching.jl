using Reactant, Test

f1(x::AbstractMatrix) = sum(x; dims=1)

f2(x::AbstractMatrix, y::Int) = x .+ y

function f3(x::AbstractArray{T,3}, y::Int) where {T}
    return Reactant.Ops.batch(Base.Fix2(f2, y), x, [1, 2])
end

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

@testset "closure" begin
    A = collect(reshape(1:30, (2, 5, 3)))
    A_ra = Reactant.to_rarray(A)

    @test @jit(f3(A_ra, 1)) ≈ A .+ 1
end
