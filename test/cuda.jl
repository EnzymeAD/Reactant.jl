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
    @cuda blocks = 1 threads = length(x) square_kernel!(x)
    return nothing
end

@testset "Square Kernel" begin
    oA = collect(1:1:64)
    A = Reactant.to_rarray(oA)
    @show @code_hlo optimize=false square!(A)
    @show @code_hlo square!(A)
    func = @compile square!(A)
    @test all(Array(A) .≈ (oA .* oA))
end
