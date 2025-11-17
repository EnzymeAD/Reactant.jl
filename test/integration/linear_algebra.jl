using LinearAlgebra, Reactant, Test

const RunningOnTPU = contains(string(Reactant.devices()[1]), "TPU")

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
    A = Reactant.TestUtils.construct_test_array(Float64, 4, 4)
    x = Reactant.TestUtils.construct_test_array(Float64, 4, 2)
    b = Reactant.TestUtils.construct_test_array(Float64, 4)

    A_ra = Reactant.to_rarray(A)
    x_ra = Reactant.to_rarray(x)
    b_ra = Reactant.to_rarray(b)

    @test @jit(muladd2(A_ra, x_ra, b_ra)) ≈ muladd2(A, x, b)
    @test @jit(muladd_5arg(A_ra, x_ra, b_ra)) ≈ muladd2(A, x, b)
    @test @jit(muladd_5arg2(A_ra, x_ra, b_ra)) ≈ 2 .* A * x .+ b
    @test @jit(A_ra * x) ≈ A * x

    @test @jit(mul_with_view1(A_ra, x_ra)) ≈ mul_with_view1(A, x)

    x2 = Reactant.TestUtils.construct_test_array(Float64, 4)
    x2_ra = Reactant.to_rarray(x2)

    @test @jit(mul_with_view2(A_ra, x2_ra)) ≈ mul_with_view2(A, x2)
    @test @jit(mul_with_view3(A_ra, x2_ra)) ≈ mul_with_view3(A, x2)

    # Mixed Precision
    x = Reactant.TestUtils.construct_test_array(Float32, 4, 2)
    x_ra = Reactant.to_rarray(x)

    @test @jit(muladd2(A_ra, x_ra, b_ra)) ≈ muladd2(A, x, b)
    @test @jit(muladd_5arg(A_ra, x_ra, b_ra)) ≈ muladd2(A, x, b)

    C_ra = similar(A_ra, Float32, size(A, 1), size(x, 2))
    C = similar(A, Float32, size(A, 1), size(x, 2))
    @jit(mul!(C_ra, A_ra, x_ra))
    mul!(C, A, x)
    @test C_ra ≈ C atol = 1e-3 rtol = 1e-2
end

@testset "triu & tril" begin
    A = Reactant.TestUtils.construct_test_array(Float64, 4, 6)
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
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 4)
    x_ra = Reactant.to_rarray(x)

    @testset for k in (-size(x, 1) + 1):(size(x, 1) - 1)
        @test @jit(diag(x_ra, k)) ≈ diag(x, k)
    end

    x = Reactant.TestUtils.construct_test_array(Float64, 4)
    x_ra = Reactant.to_rarray(x)

    @test @jit(diagm(x_ra)) ≈ diagm(x)
    @test @jit(diagm(5, 4, x_ra)) ≈ diagm(5, 4, x)
    @test @jit(diagm(4, 5, x_ra)) ≈ diagm(4, 5, x)
    @test @jit(diagm(6, 6, x_ra)) ≈ diagm(6, 6, x)
    @test_throws DimensionMismatch @jit(diagm(3, 3, x_ra))

    x1 = Reactant.TestUtils.construct_test_array(Float64, 3)
    x2 = Reactant.TestUtils.construct_test_array(Float64, 3)
    x3 = Reactant.TestUtils.construct_test_array(Float64, 2)
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
    x = Reactant.TestUtils.construct_test_array(Float64, 4, 4)
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
    @testset for T in (Int64, Float64, ComplexF32)
        @testset for (x_sz, y_sz) in [
            ((3, 4), (2, 5)), ((3, 4), (2,)), ((3,), (2, 5)), ((3,), (5,)), ((10,), ())
        ]
            x = x_sz == () ? one(T) : Reactant.TestUtils.construct_test_array(T, x_sz...)
            y = y_sz == () ? one(T) : Reactant.TestUtils.construct_test_array(T, y_sz...)
            x_ra = Reactant.to_rarray(x; track_numbers=Number)
            y_ra = Reactant.to_rarray(y; track_numbers=Number)
            @test @jit(kron(x_ra, y_ra)) ≈ kron(x, y)
        end
    end
end

@testset "axpy!" begin
    α = 3
    x = Reactant.TestUtils.construct_test_array(Int64, 4)
    x_ra = Reactant.to_rarray(x)
    y = Reactant.TestUtils.construct_test_array(Int64, 4)
    y_ra = Reactant.to_rarray(y)

    @jit axpy!(α, x_ra, y_ra)
    @test y_ra ≈ axpy!(α, x, y)

    α = 2
    x = Reactant.TestUtils.construct_test_array(Float64, 4)
    x_ra = Reactant.to_rarray(x)
    y = Reactant.TestUtils.construct_test_array(Float64, 4)
    y_ra = Reactant.to_rarray(y)

    @jit axpy!(α, x_ra, y_ra)
    @test y_ra ≈ axpy!(α, x, y)

    α = 4.12
    X = Reactant.TestUtils.construct_test_array(Float64, 3, 5)
    Y = Reactant.TestUtils.construct_test_array(Float64, 3, 5)
    X_ra = Reactant.to_rarray(X)
    Y_ra = Reactant.to_rarray(Y)

    @jit axpy!(α, X_ra, Y_ra)
    @test Y_ra ≈ axpy!(α, X, Y)

    α = 3.2 + 1im
    x = Reactant.TestUtils.construct_test_array(Complex{Float32}, 4)
    x_ra = Reactant.to_rarray(x)
    y = Reactant.TestUtils.construct_test_array(Complex{Float32}, 4)
    y_ra = Reactant.to_rarray(y)

    @jit axpy!(α, x_ra, y_ra)
    @test y_ra ≈ axpy!(α, x, y)
end

@testset "axpby!" begin
    α = 3
    β = 2
    x = Reactant.TestUtils.construct_test_array(Int64, 4)
    x_ra = Reactant.to_rarray(x)
    y = Reactant.TestUtils.construct_test_array(Int64, 4)
    y_ra = Reactant.to_rarray(y)

    @jit axpby!(α, x_ra, β, y_ra)
    @test y_ra ≈ axpby!(α, x, β, y)

    α = 2
    β = 3
    x = Reactant.TestUtils.construct_test_array(Float64, 4)
    x_ra = Reactant.to_rarray(x)
    y = Reactant.TestUtils.construct_test_array(Float64, 4)
    y_ra = Reactant.to_rarray(y)

    @jit axpby!(α, x_ra, β, y_ra)
    @test y_ra ≈ axpby!(α, x, β, y)

    α = 4.12
    X = Reactant.TestUtils.construct_test_array(Float64, 3, 5)
    Y = Reactant.TestUtils.construct_test_array(Float64, 3, 5)
    X_ra = Reactant.to_rarray(X)
    Y_ra = Reactant.to_rarray(Y)

    @jit axpby!(α, X_ra, β, Y_ra)
    @test Y_ra ≈ axpby!(α, X, β, Y)

    α = 3.2 + 1im
    β = 2.1 - 4.2im
    x = Reactant.TestUtils.construct_test_array(Complex{Float32}, 4)
    x_ra = Reactant.to_rarray(x)
    y = Reactant.TestUtils.construct_test_array(Complex{Float32}, 4)
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
        x = Reactant.TestUtils.construct_test_array(Complex{Float32}, 4)
        y = Reactant.TestUtils.construct_test_array(Complex{Float32}, 4)
        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)

        @test @jit(dot(x_ra, y_ra)) ≈ dot(x, y)

        x = Reactant.TestUtils.construct_test_array(Complex{Float32}, 2, 2)
        x_ra = Reactant.to_rarray(x)

        @test @jit(dot(x_ra, x_ra)) ≈ dot(x, x)
    end

    @testset "3-arg" begin
        x = Reactant.TestUtils.construct_test_array(Float32, 2, 2)
        y = Reactant.TestUtils.construct_test_array(Float32, 4, 5)
        z = Reactant.TestUtils.construct_test_array(Float32, 5)
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
        T == ComplexF64 && RunningOnTPU && continue

        A = Reactant.TestUtils.construct_test_array(T, 6, 6)
        B = Reactant.TestUtils.construct_test_array(T, 6, 6)
        b = Reactant.TestUtils.construct_test_array(T, 6)
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
            (T == ComplexF64 || T == Float64) && RunningOnTPU && continue

            A = rand(T, 4, 4)
            A_ra = Reactant.to_rarray(A)

            b = rand(T, 4)
            b_ra = Reactant.to_rarray(b)

            B = rand(T, 4, 3)
            B_ra = Reactant.to_rarray(B)

            @test @jit(solve_with_lu(A_ra, b_ra)) ≈ solve_with_lu(A, b) atol = 1e-4 rtol =
                1e-2
            @test @jit(solve_with_lu(A_ra, B_ra)) ≈ solve_with_lu(A, B) atol = 1e-4 rtol =
                1e-2
        end
    end

    @testset "Batched" begin
        @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
            (T == ComplexF64 || T == Float64) && RunningOnTPU && continue

            A = rand(T, 4, 4, 3, 2)
            A_ra = Reactant.to_rarray(A)

            b = rand(T, 4, 3, 2)
            b_ra = Reactant.to_rarray(b)

            B = rand(T, 4, 5, 3, 2)
            B_ra = Reactant.to_rarray(B)

            @test @jit(solve_with_lu(A_ra, b_ra)) ≈ solve_with_lu_batched(A, b) atol = 1e-4 rtol =
                1e-2
            @test @jit(solve_with_lu(A_ra, B_ra)) ≈ solve_with_lu_batched(A, B) atol = 1e-4 rtol =
                1e-2
        end
    end

    @testset "transpose! and adjoint!" begin
        A = Reactant.TestUtils.construct_test_array(Complex{Float32}, 7, 13)
        B = similar(A')::Matrix
        A_ra = Reactant.to_rarray(A)

        B_ra = Reactant.to_rarray(B)
        @test Array(@jit(transpose!(B_ra, A_ra))) ≈ transpose(A)

        B_ra = Reactant.to_rarray(B)
        @test Array(@jit(adjoint!(B_ra, A_ra))) ≈ adjoint(A)
    end

    @testset "Input Permutation" begin
        A = rand(Float32, 10, 10, 32)
        B = rand(Float32, 10, 32)
        A_ra = Reactant.to_rarray(A)
        B_ra = Reactant.to_rarray(B)

        @test @jit(solve_with_lu(A_ra, B_ra)) ≈ solve_with_lu_batched(A, B) atol = 1e-4 rtol =
            1e-2
    end
end

@testset "istriu" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 8, 8)
    x_triu = triu(x, 4)
    x_triu_ra = Reactant.to_rarray(x_triu)
    @test Bool(@jit(LinearAlgebra.istriu(x_triu_ra, 4)))
    @test Bool(@jit(LinearAlgebra.istriu(x_triu_ra, 3)))
    @test !Bool(@jit(LinearAlgebra.istriu(x_triu_ra, 5)))
end

@testset "istril" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 8, 8)
    x_tril = tril(x, -4)
    x_tril_ra = Reactant.to_rarray(x_tril)
    @test Bool(@jit(LinearAlgebra.istril(x_tril_ra, -4)))
    @test Bool(@jit(LinearAlgebra.istril(x_tril_ra, -3)))
    @test !Bool(@jit(LinearAlgebra.istril(x_tril_ra, -5)))
end

@testset "det" begin
    x_lowtri = Float32[1 0; 2 2]
    x_reg = Float32[1 -1; 2 2]
    x_uptri = Float32[1 2; 0 2]

    for x in (x_lowtri, x_reg, x_uptri)
        x_ra = Reactant.to_rarray(x)

        res_ra = @jit LinearAlgebra.logabsdet(x_ra)
        res = LinearAlgebra.logabsdet(x)
        @test res_ra[1] ≈ res[1]
        @test res_ra[2] ≈ res[2]

        res_ra = @jit LinearAlgebra.det(x_ra)
        res = LinearAlgebra.det(x)
        @test res_ra ≈ res
    end
end

@testset "inv" begin
    x_lowtri = Float32[1 0; 2 2]
    x_reg = Float32[1 -1; 2 2]
    x_uptri = Float32[1 2; 0 2]

    for x in (x_lowtri, x_reg, x_uptri)
        x_ra = Reactant.to_rarray(x)

        res_ra = @jit inv(x_ra)
        res = inv(x)
        @test res_ra ≈ res
    end
end

@testset "norm accidental promotion" begin
    x_ra = Reactant.to_rarray(rand(Float32, 4, 4))
    @test @jit(norm(x_ra)) isa ConcreteRNumber{Float32}
end

@testset "cross" begin
    x = Float32[0; 1; 0]
    x_ra = Reactant.to_rarray(x)
    y = Float32[0; 0; 1]
    y_ra = Reactant.to_rarray(y)

    @test @jit(LinearAlgebra.cross(x_ra, y_ra)) ≈ LinearAlgebra.cross(x, y)
    @test @jit(LinearAlgebra.cross(x_ra, y)) ≈ LinearAlgebra.cross(x, y)
    @test @jit(LinearAlgebra.cross(x, y_ra)) ≈ LinearAlgebra.cross(x, y)
end