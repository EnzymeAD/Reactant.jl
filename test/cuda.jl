using Reactant
using Test
using CUDA

function square_kernel!(x)
    i = threadIdx().x
    x[i] *= x[i]
    sync_threads()
    return nothing
end

# basic squaring on GPU
function square!(x)
    # @cuda blocks = 1 threads = length(x) square_kernel!(x)
    cr = @cuda launch=false square_kernel!(x)
    @show cr
    return nothing
end

@testset "Square Kernel" begin
    oA = collect(1:1:64)
    A = Reactant.to_rarray(oA)
    func = @compile square!(A)
    @test all(A .≈ (oA .* oA))
end
