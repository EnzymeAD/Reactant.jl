using Reactant
using Test
using CUDA

using Reactant_jll
@show Reactant_jll.libReactantExtra_path

function square_kernel!(x)
    #i = threadIdx().x
    #x[i] *= x[i]
    #@cuprintf("overwrote value of %f was thrown during kernel execution on thread (%d, %d, %d) in block (%d, %d, %d).\n",
    #	      0.0, threadIdx().x, threadIdx().y, threadIdx().z, blockIdx().x, blockIdx().y, blockIdx().z)
	      #x[i], threadIdx().x, threadIdx().y, threadIdx().z, blockIdx().x, blockIdx().y, blockIdx().z)

    # sync_threads()
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
    #@show @code_hlo optimize = false square!(A)
    #@show @code_hlo optimize=:before_kernel square!(A)
    #@show @code_hlo square!(A)
    func! = @compile square!(A)
    func!(A)
    @show A
    @show oA
    @test all(Array(A) .≈ (oA .* oA))
end
