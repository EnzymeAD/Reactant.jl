using Reactant
using Test
using CUDA

function square_kernel!(x, y)
    i = threadIdx().x
    x[i] *= y[i]
    sync_threads()
    return nothing
end

# basic squaring on GPU
function square!(x, y)
    @cuda blocks = 1 threads = length(x) square_kernel!(x, y)
    return nothing
end

@testset "Square Kernel" begin
    oA = collect(1:1:64)
    A = Reactant.to_rarray(oA)
    B = Reactant.to_rarray(100 .* oA)
    if CUDA.functional()
        @jit square!(A, B)
        @test all(Array(A) .≈ (oA .* oA .* 100))
        @test all(Array(B) .≈ (oA .* 100))
    else
        @code_hlo optimize = :before_kernel square!(A, B)
    end
end

function sin_kernel!(x, y)
    i = threadIdx().x
    x[i] *= sin(y[i])
    return nothing
end

# basic squaring on GPU
function sin!(x, y)
    @cuda blocks = 1 threads = length(x) sin_kernel!(x, y)
    return nothing
end

@testset "Sin Kernel" begin
    oA = collect(Float64, 1:1:64)
    A = Reactant.to_rarray(oA)
    B = Reactant.to_rarray(100 .* oA)
    if CUDA.functional()
        @jit sin!(A, B)
        @test all(Array(A) .≈ oA .* sin.(oA .* 100))
        @test all(Array(B) .≈ (oA .* 100))
    else
        @code_hlo optimize = :before_kernel sin!(A, B)
    end
end

function smul_kernel!(x, y)
    i = threadIdx().x
    x[i] *= y
    return nothing
end

# basic squaring on GPU
function smul!(x)
    @cuda blocks = 1 threads = length(x) smul_kernel!(x, 3)
    @cuda blocks = 1 threads = length(x) smul_kernel!(x, 5)
    return nothing
end

@testset "Constant Op Kernel" begin
    oA = collect(1:1:64)
    A = Reactant.to_rarray(oA)
    if CUDA.functional()
        @jit smul!(A)
        @test all(Array(A) .≈ oA .* 15)
    else
        @code_hlo optimize = :before_kernel smul!(A)
    end
end
