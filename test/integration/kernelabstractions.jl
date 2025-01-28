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

@testset "KernelAbstractions Call" begin
    A = Reactant.to_rarray(ones(100, 100))
    out = Reactant.to_rarray(ones(100, 100))
    @test all(Array(@jit(matmul!(out, A))) .≈ 100) broken = true
end

# simple square kernel
@kernel function square_kernel!(y, @Const(x))
    i = @index(Global)
    @inbounds y[i] = x[i] * x[i]
end

function square(x)
    y = similar(x)
    backend = KernelAbstractions.get_backend(x)
    kernel! = square_kernel!(backend)
    kernel!(y, x; ndrange=length(x))
    return y
end

@testset "Squaring Kernel" begin
    x = Reactant.to_rarray(collect(1:1:64) ./ 64)
    @test all(Array(@jit(square(x))) .≈ Array(x) .* Array(x))
end
