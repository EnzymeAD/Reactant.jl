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
    @test @jit(A_ra * x) ≈ A * x

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

@testset "axpy!" begin
    α = 3
    x = rand(Int64, 4)
    x_ra = Reactant.to_rarray(x)
    y = rand(Int64, 4)
    y_ra = Reactant.to_rarray(y)

    @jit axpy!(α, x_ra, y_ra)
    @test y_ra ≈ axpy!(α, x, y)

    α = 2
    x = rand(4)
    x_ra = Reactant.to_rarray(x)
    y = rand(4)
    y_ra = Reactant.to_rarray(y)

    @jit axpy!(α, x_ra, y_ra)
    @test y_ra ≈ axpy!(α, x, y)

    α = 4.12
    X = rand(3, 5)
    Y = rand(3, 5)
    X_ra = Reactant.to_rarray(X)
    Y_ra = Reactant.to_rarray(Y)

    @jit axpy!(α, X_ra, Y_ra)
    @test Y_ra ≈ axpy!(α, X, Y)

    α = 3.2 + 1im
    x = rand(Complex{Float32}, 4)
    x_ra = Reactant.to_rarray(x)
    y = rand(Complex{Float32}, 4)
    y_ra = Reactant.to_rarray(y)

    @jit axpy!(α, x_ra, y_ra)
    @test y_ra ≈ axpy!(α, x, y)
end

@testset "axpby!" begin
    α = 3
    β = 2
    x = rand(Int64, 4)
    x_ra = Reactant.to_rarray(x)
    y = rand(Int64, 4)
    y_ra = Reactant.to_rarray(y)

    @jit axpby!(α, x_ra, β, y_ra)
    @test y_ra ≈ axpby!(α, x, β, y)

    α = 2
    β = 3
    x = rand(4)
    x_ra = Reactant.to_rarray(x)
    y = rand(4)
    y_ra = Reactant.to_rarray(y)

    @jit axpby!(α, x_ra, β, y_ra)
    @test y_ra ≈ axpby!(α, x, β, y)

    α = 4.12
    X = rand(3, 5)
    Y = rand(3, 5)
    X_ra = Reactant.to_rarray(X)
    Y_ra = Reactant.to_rarray(Y)

    @jit axpby!(α, X_ra, β, Y_ra)
    @test Y_ra ≈ axpby!(α, X, β, Y)

    α = 3.2 + 1im
    β = 2.1 - 4.2im
    x = rand(Complex{Float32}, 4)
    x_ra = Reactant.to_rarray(x)
    y = rand(Complex{Float32}, 4)
    y_ra = Reactant.to_rarray(y)

    @jit axpby!(α, x_ra, β, y_ra)
    @test y_ra ≈ axpby!(α, x, β, y)
end

@testset "Dot" begin
    @testset "2-arg real" begin
        x = collect(Float32, 1:10)
        y = collect(Float32, 10:-1:1)
        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        @test @jit(dot(x_ra, y_ra)) ≈ dot(x, y)

        x = reshape(collect(Float32, 1:10), 2, 5)
        x_ra = Reactant.to_rarray(x)

        @test @jit(dot(x_ra, x_ra)) ≈ dot(x, x)
    end

    @testset "2-arg complex" begin
        x = rand(Complex{Float32}, 4)
        y = rand(Complex{Float32}, 4)
        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        @test @jit(dot(x_ra, y_ra)) ≈ dot(x, y)

        x = rand(Complex{Float32}, 2, 2)
        x_ra = Reactant.to_rarray(x)

        @test @jit(dot(x_ra, x_ra)) ≈ dot(x, x)
    end

    @testset "3-arg" begin
        x = rand(Float32, 2, 2)
        y = rand(Float32, 4, 5)
        z = rand(Float32, 5)
        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)
        z_ra = Reactant.to_rarray(z)

        @test @jit(dot(x_ra, y_ra, z_ra)) ≈ dot(x, y, z)
    end
end

@testset "Triangular ldiv and rdiv" begin
    fn1(A, b) = A \ b
    fn2(A, b) = A' \ b
    fn3(A, b) = transpose(A) \ b

    fn4(A, B) = B / A
    fn5(A, B) = B / A'
    fn6(A, B) = B / transpose(A)

    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = rand(T, 6, 6)
        B = rand(T, 6, 6)
        b = rand(T, 6)
        b_ra = Reactant.to_rarray(b)
        B_ra = Reactant.to_rarray(B)

        @testset for wT in (
            UnitLowerTriangular, UnitUpperTriangular, LowerTriangular, UpperTriangular
        )
            A_wrapped = wT(A)
            A_ra = Reactant.to_rarray(A_wrapped)

            @testset "no_tranpose" begin
                @test @jit(fn1(A_ra, b_ra)) ≈ fn1(A_wrapped, b)
                @test @jit(fn4(A_ra, B_ra)) ≈ fn4(A_wrapped, B)
            end

            @testset "adjoint" begin
                @test @jit(fn2(A_ra, b_ra)) ≈ fn2(A_wrapped, b)
                @test @jit(fn5(A_ra, B_ra)) ≈ fn5(A_wrapped, B)
            end

            @testset "transpose" begin
                @test @jit(fn3(A_ra, b_ra)) ≈ fn3(A_wrapped, b)
                @test @jit(fn6(A_ra, B_ra)) ≈ fn6(A_wrapped, B)
            end
        end
    end
end

solve_with_lu(A, b) = lu(A) \ b
function solve_with_lu_batched(A::AbstractArray{T,N}, B::AbstractArray{T,N}) where {T,N}
    A2 = reshape(A, size(A, 1), size(A, 2), prod(size(A)[3:end]))
    B2 = reshape(B, size(B, 1), size(B, 2), prod(size(B)[3:end]))
    @assert size(A2, 3) == size(B2, 3)
    return reshape(
        stack(lu(view(A2, :, :, i)) \ view(B2, :, :, i) for i in axes(A2, 3)),
        size(A2, 1),
        size(B2, 2),
        size(A)[3:end]...,
    )
end
function solve_with_lu_batched(A::AbstractArray{T,N}, b::AbstractArray{T,M}) where {T,N,M}
    @assert N == M + 1
    B = reshape(b, size(b, 1), 1, size(b)[2:end]...)
    return dropdims(solve_with_lu_batched(A, B); dims=2)
end

@testset "LU Factorization" begin
    @testset "Un-batched" begin
        @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
            A = rand(T, 4, 4)
            A_ra = Reactant.to_rarray(A)

            b = rand(T, 4)
            b_ra = Reactant.to_rarray(b)

            B = rand(T, 4, 3)
            B_ra = Reactant.to_rarray(B)

            @test @jit(solve_with_lu(A_ra, b_ra)) ≈ solve_with_lu(A, b)
            @test @jit(solve_with_lu(A_ra, B_ra)) ≈ solve_with_lu(A, B)
        end
    end

    @testset "Batched" begin
        @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
            A = rand(T, 4, 4, 3, 2)
            A_ra = Reactant.to_rarray(A)

            b = rand(T, 4, 3, 2)
            b_ra = Reactant.to_rarray(b)

            B = rand(T, 4, 5, 3, 2)
            B_ra = Reactant.to_rarray(B)

            @test @jit(solve_with_lu(A_ra, b_ra)) ≈ solve_with_lu_batched(A, b)
            @test @jit(solve_with_lu(A_ra, B_ra)) ≈ solve_with_lu_batched(A, B)
        end
    end

    @testset "Input Permutation" begin
        A = rand(Float32, 10, 10, 32)
        B = rand(Float32, 10, 32)
        A_ra = Reactant.to_rarray(A)
        B_ra = Reactant.to_rarray(B)

        @test @jit(solve_with_lu(A_ra, B_ra)) ≈ solve_with_lu_batched(A, B)
    end
end
