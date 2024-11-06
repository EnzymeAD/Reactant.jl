using LinearAlgebra, Reactant

function muladd2(A, x, b)
    C = similar(A, promote_type(eltype(A), eltype(b)), size(A, 1), size(x, 2))
    mul!(C, A, x)
    C .+= b
    return C
end

@testset begin
    A = Reactant.to_rarray(rand(4, 4))
    x = Reactant.to_rarray(rand(4, 2))
    b = Reactant.to_rarray(rand(4))

    @test @jit(muladd2(A, x, b)) ≈ muladd2(A, x, b)

    # Mixed Precision
    x = Reactant.to_rarray(rand(Float32, 4, 2))

    @test @jit(muladd2(A, x, b)) ≈ muladd2(A, x, b)

    C = similar(A, Float32, size(A, 1), size(x, 2))
    @jit(mul!(C, A, x))
    @test C ≈ A * x
end
