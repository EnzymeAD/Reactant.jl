using LinearAlgebra, Reactant, Test

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

@testset "diag / diagm" begin
    x = rand(2, 4)
    x_ra = Reactant.to_rarray(x)

    @testset for k in (-size(x, 1) + 1):(size(x, 1) - 1)
        @test @jit(diag(x_ra, k)) ≈ diag(x, k)
    end

    x = rand(4)
    x_ra = Reactant.to_rarray(x)

    @test @jit(diagm(x_ra)) ≈ diagm(x)
    @test @jit(diagm(5, 4, x_ra)) ≈ diagm(5, 4, x)
    @test @jit(diagm(4, 5, x_ra)) ≈ diagm(4, 5, x)
    @test @jit(diagm(6, 6, x_ra)) ≈ diagm(6, 6, x)
    @test_throws DimensionMismatch @jit(diagm(3, 3, x_ra))

    x1 = rand(3)
    x2 = rand(3)
    x3 = rand(2)
    x_ra1 = Reactant.to_rarray(x1)
    x_ra2 = Reactant.to_rarray(x2)
    x_ra3 = Reactant.to_rarray(x3)

    @test @jit(diagm(1 => x_ra1)) ≈ diagm(1 => x1)
    @test @jit(diagm(1 => x_ra1, -1 => x_ra3)) ≈ diagm(1 => x1, -1 => x3)
    @test @jit(diagm(1 => x_ra1, 1 => x_ra2)) ≈ diagm(1 => x1, 1 => x2)
end

# TODO: Currently <Wrapper Type>(x) * x goes down the generic matmul path but it should
#       clearly be optimized
mul_diagonal(x) = Diagonal(x) * x
mul_tridiagonal(x) = Tridiagonal(x) * x
mul_unit_lower_triangular(x) = UnitLowerTriangular(x) * x
mul_unit_upper_triangular(x) = UnitUpperTriangular(x) * x
mul_lower_triangular(x) = LowerTriangular(x) * x
mul_upper_triangular(x) = UpperTriangular(x) * x
mul_symmetric(x) = Symmetric(x) * x

@testset "Wrapper Types Matrix Multiplication" begin
    x = rand(4, 4)
    x_ra = Reactant.to_rarray(x)

    @testset "$(wrapper_type)" for (wrapper_type, fn) in [
        (Diagonal, mul_diagonal),
        (Tridiagonal, mul_tridiagonal),
        (UnitLowerTriangular, mul_unit_lower_triangular),
        (UnitUpperTriangular, mul_unit_upper_triangular),
        (LowerTriangular, mul_lower_triangular),
        (UpperTriangular, mul_upper_triangular),
        (Symmetric, mul_symmetric),
    ]
        @test @jit(fn(x_ra)) ≈ fn(x)
    end
end

@testset "kron" begin
    @testset for T in (Int64, Float64, ComplexF64)
        @testset for (x_sz, y_sz) in [
            ((3, 4), (2, 5)), ((3, 4), (2,)), ((3,), (2, 5)), ((3,), (5,)), ((10,), ())
        ]
            x = x_sz == () ? rand(T) : rand(T, x_sz)
            y = y_sz == () ? rand(T) : rand(T, y_sz)
            x_ra = Reactant.to_rarray(x; track_numbers=Number)
            y_ra = Reactant.to_rarray(y; track_numbers=Number)
            @test @jit(kron(x_ra, y_ra)) ≈ kron(x, y)
        end
    end
end

@testset "dot" begin
    a = [1, 2, 3, 4]
    b = [-2, 5, 6, 7]
    a_ra = Reactant.to_rarray(a)
    b_ra = Reactant.to_rarray(b)

    @test @jit dot(a_ra, b_ra) ≈ dot(a, b)

    a = rand(4)
    b = rand(4)
    a_ra = Reactant.to_rarray(a)
    b_ra = Reactant.to_rarray(b)
    
    @test @jit dot(a_ra, b_ra) ≈ dot(a, b)

    a = rand(Complex{Float64}, 4)
    b = rand(Complex{Float64}, 4)
    a_ra = Reactant.to_rarray(a)
    b_ra = Reactant.to_rarray(b)
    ab_ra = @jit dot(a_ra, b_ra)
    ab = dot(a,b)
    
    @test ab_ra ≈ ab 
    
    # I found this strange. If in dot_generat I did not apply Ops.conj,
    # calling the line below still gives test passed while the one above does not
    # @test @jit dot(a_ra, b_ra) ≈ dot(a, b)
end


@testset "cross" begin
    a = [1, 2, 3]
    b = [-2, 5, 7]
    a_ra = Reactant.to_rarray(a)
    b_ra = Reactant.to_rarray(b)

    @test @jit cross(a_ra, b_ra) ≈ cross(a, b)

    a = rand(3)
    b = rand(3)
    a_ra = Reactant.to_rarray(a)
    b_ra = Reactant.to_rarray(b)
    
    @test @jit dot(a_ra, b_ra) ≈ dot(a, b)

    a = rand(Complex{Float64}, 3)
    b = rand(Complex{Float64}, 3)
    a_ra = Reactant.to_rarray(a)
    b_ra = Reactant.to_rarray(b)

    @test @jit cross(a_ra, b_ra) ≈ cross(a,b)
end    

