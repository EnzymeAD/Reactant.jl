using CUDA, KernelAbstractions, Reactant

# Simple kernel for matrix multiplication
@kernel function matmul_kernel!(output, a)
    i, j = @index(Global, NTuple)
    # creating a temporary sum variable for matrix multiplication

    tmp_sum = zero(eltype(output))
    for k in 1:size(a)[2]
        @inbounds tmp_sum += a[i, k] * a[k, j]
    end

    @inbounds output[i, j] = tmp_sum
end

# Creating a wrapper kernel for launching with error checks
function matmul!(output, a)
    backend = KernelAbstractions.get_backend(output)
    kernel! = matmul_kernel!(backend)
    kernel!(output, a; ndrange=size(output))
    return KernelAbstractions.synchronize(backend)
end

# https://github.com/EnzymeAD/Reactant.jl/issues/614
const skip_non_cuda_tests = true

@static if !Sys.isapple()
    @testset "KernelAbstractions Matmul" begin
        A = Reactant.to_rarray(ones(100, 100))
        out = Reactant.to_rarray(ones(100, 100))
        if CUDA.functional()
            @test all(Array(@jit(matmul!(out, A))) .≈ 100) broken = true
        else
            @static if skip_non_cuda_tests
                @test false broken = true
            else
                @code_hlo optimize = :before_kernel matmul!(out, A)
            end
        end
    end
end
