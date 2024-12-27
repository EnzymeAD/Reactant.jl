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
    @jit square!(A, B)
    @test all(Array(A) .≈ (oA .* oA .* 100))
    @test all(Array(B) .≈ (oA .* 100))
end
