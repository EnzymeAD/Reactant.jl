using Reactant, Test, CUDA

const ReactantCUDAExt = Base.get_extension(Reactant, :ReactantCUDAExt)

@testset "Promote CuTraced" begin
    TFT = ReactantCUDAExt.CuTracedRNumber{Float64,1}
    FT = Float64
    @test Reactant.promote_traced_type(TFT, FT) == TFT
    @test Base.promote_type(TFT, FT) == FT
end

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
    A = Reactant.to_rarray(fill(1))
    @jit tuplef(A)
    @test all(Array(A) .≈ 3)

    A = Reactant.to_rarray(fill(1))
    @jit tuplef2(A)
    @test all(Array(A) .≈ 5)
end
A = Reactant.to_rarray(fill(1))
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
    a = Reactant.to_rarray([3])

    @jit aliased(a)
    @test all(Array(a) .== 9)
end

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
        b = Reactant.to_rarray([3])
        @jit mixed(a, b)
        @test all(Array(a) .== 4)
        @test all(Array(b) .== 12)
    end
end

function mul_number_kernel!(x, y)
    i = threadIdx().x
    x[i] *= y
    return nothing
end

function mul_number!(x, y)
    @cuda blocks = 1 threads = length(x) mul_number_kernel!(x, y)
    return nothing
end

@testset "Mul Number" begin
    oA = collect(Float64, 1:1:64)
    A = Reactant.to_rarray(oA)
    B = ConcreteRNumber(3.1)
    @jit mul_number!(A, B)
    @test all(Array(A) .≈ oA .* 3.1)
end

function pow_number_kernel!(x, y)
    i = threadIdx().x
    x[i] *= (y^2)
    return nothing
end

function pow_number!(x, y)
    @cuda blocks = 1 threads = length(x) pow_number_kernel!(x, y)
    return nothing
end

@testset "Pow Number" begin
    oA = collect(Float64, 1:1:64)
    A = Reactant.to_rarray(oA)
    B = ConcreteRNumber(3.1)
    @jit pow_number!(A, B)
    @test all(Array(A) .≈ oA .* 3.1 * 3.1)
end

function searchsorted_kernel!(x, y)
    i = threadIdx().x
    times = 0:0.01:4.5
    z = searchsortedfirst(times, y)
    x[i] = z
    return nothing
end

function searchsorted!(x, y)
    @cuda blocks = 1 threads = length(x) searchsorted_kernel!(x, y)
    return nothing
end

@testset "Search sorted" begin
    oA = collect(Float64, 1:1:64)
    A = Reactant.to_rarray(oA)
    B = ConcreteRNumber(3.1)
    @test begin
        @jit searchsorted!(A, B)
        all(Array(A) .≈ 311)
    end broken = contains(string(Reactant.devices()[1]), "TPU")
end

function convert_mul_kernel!(Gu, w::FT) where {FT}
    r = FT(0.5) * w
    @inbounds Gu[1, 1, 1] = r
    return nothing
end

function convert_mul!(Gu, w)
    @cuda blocks = 1 threads = 1 convert_mul_kernel!(Gu, w)
    return nothing
end

@testset "Convert mul" begin
    w = Reactant.ConcreteRNumber(0.6)
    Gu = Reactant.to_rarray(ones(24, 24, 24))
    @jit convert_mul!(Gu, w)
    Gui = Array((Gu))
    @test Gui[1] ≈ 0.3
end
