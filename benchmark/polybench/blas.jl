using Reactant
Reactant.allowscalar(true)

include("common.jl")

# TODO: we are having trouble raising symm.
#       See https://github.com/EnzymeAD/Enzyme-JAX/issues/1864
# TODO: For supporting trmm we need to support non-static loop starts/limits

function gemm(
    alpha::T, beta::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}
) where {T}
    @trace for i in axes(A, 1)
        @trace for j in axes(B, 2)
            C[i, j] *= beta
        end
        @trace for k in axes(A, 2)
            @trace for j in axes(B, 2)
                C[i, j] += alpha * A[i, k] * B[k, j]
            end
        end
    end
    return C
end

function run_gemm_benchmark!(results, backend)
    N, M = 2048, 4096
    α = 2.0f0
    β = 3.0f0
    A = rand(Float32, N, M)
    B = rand(Float32, M, N)
    C = rand(Float32, N, N)

    run_benchmark!(
        results,
        "gemm [$(N), $(M)]",
        backend,
        gemm,
        (α, β, A, B, C),
        (α, β, A, B, C) -> α .* A * B .+ β .* C;
        track_numbers=true,
    )
    return nothing
end

function gemmver(
    alpha::T,
    beta::T,
    u1::AbstractVector{T},
    u2::AbstractVector{T},
    v1::AbstractVector{T},
    v2::AbstractVector{T},
    A::AbstractMatrix{T},
    y::AbstractVector{T},
    z::AbstractVector{T},
) where {T}
    x = zeros(T, axes(A, 1))
    w = zeros(T, axes(A, 1))

    @trace for i in axes(A, 1)
        @trace for j in axes(A, 2)
            A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]
        end
    end

    @trace for i in axes(A, 1)
        @trace for j in axes(A, 2)
            x[i] = x[i] + beta * A[j, i] * y[j]
        end
    end

    @trace for i in axes(A, 1)
        x[i] = x[i] + z[i]
    end

    @trace for i in axes(A, 1)
        @trace for j in axes(A, 2)
            w[i] = w[i] + alpha * A[i, j] * x[j]
        end
    end

    return A, x, w
end

function gemmver_vectorized(
    alpha::T,
    beta::T,
    u1::AbstractVector{T},
    u2::AbstractVector{T},
    v1::AbstractVector{T},
    v2::AbstractVector{T},
    A::AbstractMatrix{T},
    y::AbstractVector{T},
    z::AbstractVector{T},
) where {T}
    Â = A .+ u1 .* v1' .+ u2 .* v2'
    x = beta .* Â' * y .+ z
    w = alpha .* Â * x
    return Â, x, w
end

function run_gemmver_benchmark!(results, backend)
    N = 2048

    α = 0.0001f0
    β = 0.05f0
    u1 = rand(Float32, N)
    u2 = rand(Float32, N)
    v1 = rand(Float32, N)
    v2 = rand(Float32, N)
    A = rand(Float32, N, N)
    y = rand(Float32, N)
    z = rand(Float32, N)

    run_benchmark!(
        results,
        "gemmver [$(N)]",
        backend,
        gemmver,
        (α, β, u1, u2, v1, v2, A, y, z),
        gemmver_vectorized;
        track_numbers=true,
    )
    return nothing
end

function gesummv(
    alpha::T, beta::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, x::AbstractVector{T}
) where {T}
    tmp = similar(x)
    y = similar(x)
    @trace for i in axes(A, 1)
        tmp[i] = zero(eltype(A))
        y[i] = zero(eltype(A))
        @trace for j in axes(A, 2)
            tmp[i] = A[i, j] * x[j] + tmp[i]
            y[i] = B[i, j] * x[j] + y[i]
        end
        y[i] = alpha * tmp[i] + beta * y[i]
    end
    return y
end

function gesummv_vectorized(
    α::T, β::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, x::AbstractVector{T}
) where {T}
    return (α .* A .+ β .* B) * x
end

function run_gesummv_benchmark!(results, backend)
    N = 4096

    α = 2.0f0
    β = 3.0f0
    A = rand(Float32, N, N)
    B = rand(Float32, N, N)
    x = rand(Float32, N)

    run_benchmark!(
        results,
        "gesummv [$(N)]",
        backend,
        gesummv,
        (α, β, A, B, x),
        gesummv_vectorized;
        track_numbers=true,
    )
    return nothing
end

# NOTE: we make a small change to this to allow static loop bounds
function syr2k(
    alpha::T, beta::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}
) where {T}
    @trace for i in axes(A, 1)
        @trace for j in axes(C, 2)
            C[i, j] *= beta
        end
        @trace for k in axes(A, 2)
            @trace for j in axes(A, 1)
                C[i, j] += A[j, k] * alpha * B[i, k] + B[j, k] * alpha * A[i, k]
            end
        end
    end
    return C
end

function syr2k_vectorized(
    α::T, β::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}
) where {T}
    return α .* A * B' + α .* B * A' + β .* C
end

function run_syr2k_benchmark!(results, backend)
    N = 2048

    α = 2.0f0
    β = 3.0f0
    A = rand(Float32, N, N)
    A = A + A' # symmetric
    B = rand(Float32, N, N)
    C = rand(Float32, N, N)
    C = C + C' # symmetric

    run_benchmark!(
        results,
        "syr2k [$(N)]",
        backend,
        syr2k,
        (α, β, A, B, C),
        syr2k_vectorized;
        track_numbers=true,
    )
    return nothing
end

function syrk(alpha::T, beta::T, A::AbstractMatrix{T}, C::AbstractMatrix{T}) where {T}
    @trace for i in axes(A, 1)
        @trace for j in axes(C, 2)
            C[i, j] *= beta
        end
        @trace for k in axes(A, 2)
            @trace for j in axes(A, 1)
                C[i, j] += A[j, k] * alpha * A[i, k]
            end
        end
    end
    return C
end

function syrk_vectorized(α::T, β::T, A::AbstractMatrix{T}, C::AbstractMatrix{T}) where {T}
    return (α .* A * A') + (β .* C)
end

function run_syrk_benchmark!(results, backend)
    N = 2048

    α = 2.0f0
    β = 3.0f0
    A = rand(Float32, N, N)
    A = A + A' # symmetric
    C = rand(Float32, N, N)
    C = C + C' # symmetric

    run_benchmark!(
        results,
        "syrk [$(N)]",
        backend,
        syrk,
        (α, β, A, C),
        syrk_vectorized;
        track_numbers=true,
    )
    return nothing
end

function run_blas_benchmarks!(results, backend)
    run_gemm_benchmark!(results, backend)
    run_gemmver_benchmark!(results, backend)
    run_gesummv_benchmark!(results, backend)
    run_syr2k_benchmark!(results, backend)
    run_syrk_benchmark!(results, backend)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    backend = get_backend()
    results = Dict()
    run_blas_benchmarks!(results, backend)
end
