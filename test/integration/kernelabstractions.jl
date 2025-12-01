using CUDA, KernelAbstractions, Reactant, Test

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

@testset "KernelAbstractions Matmul" begin
    A = Reactant.to_rarray(ones(100, 100))
    out = Reactant.to_rarray(ones(100, 100))
    platform_name = Reactant.XLA.platform_name(Reactant.XLA.default_backend())
    raise = platform_name ∉ ("cpu", "cuda")
    @jit raise = raise matmul!(out, A)
    out_c = Array(out)
    A_c = Array(A)
    @test out_c ≈ A_c * A_c'
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

@testset "KernelAbstractions Square" begin
    x = Reactant.to_rarray(collect(1:1:64) ./ 64)

    platform_name = Reactant.XLA.platform_name(Reactant.XLA.default_backend())
    raise = platform_name ∉ ("cpu", "cuda")

    b = get_backend(x)
    @test b isa Base.get_extension(Reactant, :ReactantKernelAbstractionsExt).ReactantBackend
    let y = allocate(b, Float32, (100, 10))
        @test y isa ConcreteRArray{Float32,2}
        @test size(y) == (100, 10)
    end
    let y = KernelAbstractions.zeros(b, Float32, (100, 10))
        @test y isa ConcreteRArray{Float32,2}
        @test Array(y) == zeros(Float32, 100, 10)
    end
    let y = KernelAbstractions.ones(b, Float32, (100, 10))
        @test y isa ConcreteRArray{Float32,2}
        @test Array(y) == ones(Float32, 100, 10)
    end

    @test all(Array(@jit(raise = raise, square(x))) .≈ Array(x) .* Array(x))
end
