using CUDA, KernelAbstractions, Reactant, Test, FileCheck

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

# TODO(#2253): raising fails on TPU CI.
#       https://github.com/EnzymeAD/Reactant.jl/pull/1923#discussion_r2580461294
if !Reactant.Accelerators.TPU.has_tpu()
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
end

# simple square kernel
@kernel function square_kernel!(y, @Const(x))
    i = @index(Global)
    @inbounds y[i] = x[i] * x[i]
end

@kernel function weno_weights_kernel!(out, @Const(c))
    i = @index(Global)
    @inbounds begin
        α1 = (2 / 3) / c[i]
        α2 = (1 / 3) / c[i + 1]

        out[i] = α1 + α2   # bare sum of the two lanes
    end
end

function square(x)
    y = similar(x)
    backend = KernelAbstractions.get_backend(x)
    kernel! = square_kernel!(backend)
    kernel!(y, x; ndrange=length(x))
    return y
end

function run_weno!(out, c)
    backend = KernelAbstractions.get_backend(out)
    weno_weights_kernel!(backend)(out, c; ndrange=length(out))
    return KernelAbstractions.synchronize(backend)
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

@testset "KernelAbstractions WENO weights" begin
    N = 64
    c = Reactant.to_rarray(sin.((1:(N + 2)) ./ 3.0))
    out = Reactant.to_rarray(zeros(N))

    compiled! = Reactant.@compile raise = true run_weno!(out, c)
    compiled!(out, c)

    c_cpu = sin.((1:(N + 2)) ./ 3.0)
    expected = (2 / 3) ./ c_cpu[1:N] .+ (1 / 3) ./ c_cpu[2:(N + 1)]
    @test Array(out) ≈ expected
end

@kernel function fma_kernel!(out, @Const(a), @Const(b), @Const(c))
    i = @index(Global)
    @inbounds out[i] = fma(a[i], b[i], c[i])
end

function run_fma!(out, a, b, c)
    backend = KernelAbstractions.get_backend(out)
    kernel! = fma_kernel!(backend)
    kernel!(out, a, b, c; ndrange=length(out))
    return out
end

@testset "Compile FMA" begin
    a = Reactant.to_rarray(Float64[1.0, 2.0, 3.0, 4.0])
    b = Reactant.to_rarray(Float64[2.0, 3.0, 4.0, 5.0])
    c = Reactant.to_rarray(Float64[0.5, 0.5, 0.5, 0.5])
    out = Reactant.to_rarray(zeros(Float64, 4))

    ir = repr(Reactant.@code_hlo raise = true run_fma!(out, a, b, c))
    @test @filecheck begin
        @check "%0 = stablehlo.multiply %arg1, %arg2 : tensor<4xf64>"
        @check_next "%1 = stablehlo.add %0, %arg3 : tensor<4xf64>"
        @check_next "return %1 : tensor<4xf64>"
        ir
    end
end
