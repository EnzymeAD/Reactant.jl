
using Reactant
using Test
using CUDA

function square_kernel!(x, y)
    i = threadIdx().x
    x[i] *= y[i]
    # We don't yet auto lower this via polygeist
    # sync_threads()
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
    @jit square!(A, B)
    @test all(Array(A) .≈ (oA .* oA .* 100))
    @test all(Array(B) .≈ (oA .* 100))
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
    @jit sin!(A, B)
    @test all(Array(A) .≈ oA .* sin.(oA .* 100))
    @test all(Array(B) .≈ (oA .* 100))
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
    @jit smul!(A)
    @test all(Array(A) .≈ oA .* 15)
end

function tuplef!(tup)
    tup[1][] += 2
    return nothing
end

function tuplef2!(tup)
    tup[2][] *= tup[1]
    return nothing
end

tuplef(a) = @cuda threads = 1 tuplef!((a,))
tuplef2(a) = @cuda threads = 1 tuplef2!((5, a))

@testset "Structured Kernel Arguments" begin
    A = ConcreteRArray(fill(1))
    @jit tuplef(A)
    @test all(Array(A) .≈ 3)

    A = ConcreteRArray(fill(1))
    @jit tuplef2(A)
    @test all(Array(A) .≈ 5)
end
A = ConcreteRArray(fill(1))
@jit tuplef2(A)
@test all(Array(A) .≈ 5)


# TODO this same code fails if we use a 0-d array...?
# maybe weird cuda things
function aliased!(tup)
    x, y = tup
    x[1] *= y[1]
    return nothing
end

function aliased(s)
    tup = (s, s)
    @cuda threads = 1 aliased!(tup)
    return nothing
end

@testset "Aliasing arguments" begin
    a = ConcreteRArray([3])

    @jit aliased(a)
    @test all(Array(a) .== 9)
end

using Reactant, CUDA

function cmul!(a, b)
    b[1] *= a[1]
    return nothing
end

function mixed(a, b)
    @cuda threads = 1 cmul!(a, b)
    return nothing
end

@testset "Non-traced argument" begin
    if CUDA.functional()
        a = CuArray([4])
        b = ConcreteRArray([3])
        @jit mixed(a, b)
        @test all(Array(a) .== 4)
        @test all(Array(b) .== 12)
    end
end
