using LinearAlgebra, Reactant

function muladd2(A, x, b)
    C = similar(A, promote_type(eltype(A), eltype(b)), size(A, 1), size(x, 2))
    mul!(C, A, x)
    C .+= b
    return C
end

function muladd_5arg(A, x, b)
    C = similar(A, promote_type(eltype(A), eltype(b)), size(A, 1), size(x, 2))
    C .= b
    mul!(C, A, x, 1, 1)
    return C
end

function muladd_5arg2(A, x, b)
    C = similar(A, promote_type(eltype(A), eltype(b)), size(A, 1), size(x, 2))
    C .= b
    mul!(C, A, x, 2.0f0, 1)
    return C
end

function mul_with_view1(A, x)
    B = view(A, 1:2, 1:2)
    x = view(x, 1:2, :)
    C = similar(B, promote_type(eltype(A), eltype(x)), size(B, 1), size(x, 2))
    mul!(C, B, x)
    return C
end

function mul_with_view2(A, x)
    B = view(A, 1:2, 1:2)
    x = view(x, 1:2)
    C = similar(B, promote_type(eltype(A), eltype(x)), size(B, 1), size(x, 2))
    mul!(C, B, x)
    return C
end

function mul_with_view3(A, x)
    B = view(A, 1:2, 1:2)
    x = view(x, 1:2)
    C = similar(B, promote_type(eltype(A), eltype(x)), size(B, 1))
    mul!(C, B, x)
    return C
end

@testset "Matrix Multiplication" begin
    A = rand(4, 4)
    x = rand(4, 2)
    b = rand(4)

    A_ra = Reactant.to_rarray(A)
    x_ra = Reactant.to_rarray(x)
    b_ra = Reactant.to_rarray(b)

    @test @jit(muladd2(A_ra, x_ra, b_ra)) ≈ muladd2(A, x, b)
    @test @jit(muladd_5arg(A_ra, x_ra, b_ra)) ≈ muladd2(A, x, b)
    @test @jit(muladd_5arg2(A_ra, x_ra, b_ra)) ≈ 2 .* A * x .+ b

    @test @jit(mul_with_view1(A_ra, x_ra)) ≈ mul_with_view1(A, x)

    x2 = rand(4)
    x2_ra = Reactant.to_rarray(x2)

    @test @jit(mul_with_view2(A_ra, x2_ra)) ≈ mul_with_view2(A, x2)
    @test @jit(mul_with_view3(A_ra, x2_ra)) ≈ mul_with_view3(A, x2)

    # Mixed Precision
    x = rand(Float32, 4, 2)
    x_ra = Reactant.to_rarray(x)

    @test @jit(muladd2(A_ra, x_ra, b_ra)) ≈ muladd2(A, x, b)
    @test @jit(muladd_5arg(A_ra, x_ra, b_ra)) ≈ muladd2(A, x, b)

    C_ra = similar(A_ra, Float32, size(A, 1), size(x, 2))
    @jit(mul!(C_ra, A_ra, x_ra))
    @test C_ra ≈ A * x
end

@testset "triu & tril" begin
    A = rand(4, 6)
    A_ra = Reactant.to_rarray(A)

    @test @jit(triu(A_ra)) ≈ triu(A)
    @test @jit(tril(A_ra)) ≈ tril(A)
    @test @jit(triu(A_ra, 2)) ≈ triu(A, 2)
    @test @jit(tril(A_ra, 2)) ≈ tril(A, 2)
    @test @jit(triu(A_ra, -1)) ≈ triu(A, -1)
    @test @jit(tril(A_ra, -1)) ≈ tril(A, -1)

    A_ra = Reactant.to_rarray(A)
    @jit(triu!(A_ra))
    @test A_ra ≈ triu(A)

    A_ra = Reactant.to_rarray(A)
    @jit(tril!(A_ra))
    @test A_ra ≈ tril(A)

    A_ra = Reactant.to_rarray(A)
    @jit(triu!(A_ra, 2))
    @test A_ra ≈ triu(A, 2)

    A_ra = Reactant.to_rarray(A)
    @jit(tril!(A_ra, 2))
    @test A_ra ≈ tril(A, 2)

    A_ra = Reactant.to_rarray(A)
    @jit(triu!(A_ra, -1))
    @test A_ra ≈ triu(A, -1)

    A_ra = Reactant.to_rarray(A)
    @jit(tril!(A_ra, -1))
    @test A_ra ≈ tril(A, -1)
end
