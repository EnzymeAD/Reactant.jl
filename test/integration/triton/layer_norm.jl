using PythonCall, Reactant, Test

pyimport("sys").path.append(@__DIR__)

layer_norm_kernel = pyimport("layer_norm").layer_norm_fwd_fused
layer_norm_kernel_v2 = pyimport("layer_norm").layer_norm_fwd_fused_simple

const RunningOnCUDA = contains(string(Reactant.devices()[1]), "CUDA")

function layer_norm_triton(
    x::AbstractMatrix{T}, weight::AbstractVector{T}, bias::AbstractVector{T}, simple::Bool
) where {T}
    x_transposed = permutedims(x, (2, 1)) # match python array layout
    y = similar(x_transposed)
    M, N = size(x_transposed)
    mean = similar(x_transposed, Float32, M)
    rstd = similar(x_transposed, Float32, M)

    max_fused_size = 65536 ÷ sizeof(T)
    block_size = min(max_fused_size, nextpow(2, N))

    if N > block_size
        throw(ArgumentError("This layer norm doesn't support feature dim >= 64KB."))
    end

    (simple ? layer_norm_kernel_v2 : layer_norm_kernel)(
        x_transposed,
        y,
        weight,
        bias,
        mean,
        rstd,
        Reactant.rowmajor_stride(x_transposed, 1),
        N,
        1.0f-5,
        block_size;
        num_warps=min(max(block_size ÷ 256, 1), 8),
        num_ctas=1,
        grid=(M,),
    )

    return permutedims(y, (2, 1)), mean, rstd
end

function layer_norm_naive(
    x::AbstractMatrix{T}, weight::AbstractVector{T}, bias::AbstractVector{T}
) where {T}
    mean = sum(x; dims=1) ./ size(x, 1)
    rstd = 1 ./ sqrt.(sum(abs2, x .- mean; dims=1) ./ size(x, 1) .+ 1e-5)
    x_hat = (x .- mean) .* rstd
    return x_hat .* weight .+ bias, vec(mean), vec(rstd)
end

@testset "fused_layer_norm" begin
    if RunningOnCUDA
        x_ra = Reactant.to_rarray(rand(Float32, 257, 2056))
        weight_ra = Reactant.to_rarray(rand(Float32, 257))
        bias_ra = Reactant.to_rarray(rand(Float32, 257))

        y_ra1, mean_ra1, rstd_ra1 = @jit layer_norm_triton(x_ra, weight_ra, bias_ra, false)
        y_ra2, mean_ra2, rstd_ra2 = @jit layer_norm_naive(x_ra, weight_ra, bias_ra)
        y_ra3, mean_ra3, rstd_ra3 = @jit layer_norm_triton(x_ra, weight_ra, bias_ra, true)

        @test y_ra1 ≈ y_ra2
        @test y_ra2 ≈ y_ra3
        @test mean_ra1 ≈ mean_ra2
        @test mean_ra2 ≈ mean_ra3
        @test rstd_ra1 ≈ rstd_ra2
        @test rstd_ra2 ≈ rstd_ra3
    end
end
