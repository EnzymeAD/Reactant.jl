using CUDA
using KernelAbstractions
using Reactant

using CUDA: CuArray

# Simple kernel for matrix multiplication
@kernel function matmul_kernel!(output, a)
    i, j = @index(Global, NTuple)
    # creating a temporary sum variable for matrix multiplication

    tmp_sum = zero(eltype(output))
    for k in 1:size(a)[2]
        tmp_sum += a[i, k] * a[k, j]
    end

    return output[i, j] = tmp_sum
end

# Creating a wrapper kernel for launching with error checks
function matmul!(output, a, backend)
    kernel! = matmul_kernel!(backend)
    kernel!(output, a; ndrange=size(output))
    return KernelAbstractions.synchronize(backend)
end

@testset "KernelAbstractions Call" begin
    backend = KernelAbstractions.get_backend(CuArray(ones(1)))
    A = Reactant.to_rarray(CuArray(ones(100, 100)))
    out = Reactant.to_rarray(CuArray(ones(100, 100)))
    @jit matmul!(out, A, backend)
    @test all(Array(out) .≈ 100)
end
