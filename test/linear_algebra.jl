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

function  mul_with_view2(A, x)
    B = view(A, 1:2, 1:2)
    x = view(x, 1:2)
    C = similar(B, promote_type(eltype(A), eltype(x)), size(B, 1), size(x, 2))
    mul!(C, B, x)
    return C
end

function  mul_with_view3(A, x)
    B = view(A, 1:2, 1:2)
    x = view(x, 1:2)
    C = similar(B, promote_type(eltype(A), eltype(x)), size(B, 1))
    mul!(C, B, x)
    return C
end

@testset begin
    A = Reactant.to_rarray(rand(4, 4))
    x = Reactant.to_rarray(rand(4, 2))
    b = Reactant.to_rarray(rand(4))

    @test @jit(muladd2(A, x, b)) ≈ muladd2(A, x, b)
    @test @jit(muladd_5arg(A, x, b)) ≈ muladd2(A, x, b)
    @test @jit(muladd_5arg2(A, x, b)) ≈ 2 .* A * x .+ b

    @test @jit(mul_with_view1(A, x)) ≈ mul_with_view1(A, x)

    x2 = Reactant.to_rarray(rand(4))
    @test @jit(mul_with_view2(A, x2)) ≈ mul_with_view2(A, x2)
    @test @jit(mul_with_view3(A, x2)) ≈ mul_with_view3(A, x2)

    # Mixed Precision
    x = Reactant.to_rarray(rand(Float32, 4, 2))

    @test @jit(muladd2(A, x, b)) ≈ muladd2(A, x, b)
    @test @jit(muladd_5arg(A, x, b)) ≈ muladd2(A, x, b)

    C = similar(A, Float32, size(A, 1), size(x, 2))
    @jit(mul!(C, A, x))
    @test C ≈ A * x
end
