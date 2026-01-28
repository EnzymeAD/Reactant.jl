using Reactant
Reactant.allowscalar(true)

include("common.jl")

function kernel_2mm(
    alpha::T,
    beta::T,
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    C::AbstractMatrix{T},
    D::AbstractMatrix{T},
) where {T}
    tmp = similar(A, T, size(A, 1), size(B, 2))
    @trace for i in axes(A, 1)
        @trace for j in axes(B, 2)
            tmp[i, j] = 0
            @trace for k in axes(A, 2)
                tmp[i, j] += alpha * A[i, k] * B[k, j]
            end
        end
    end
    @trace for i in axes(A, 1)
        @trace for j in axes(C, 2)
            D[i, j] *= beta
            @trace for k in axes(B, 1)
                D[i, j] += tmp[i, k] * C[k, j]
            end
        end
    end
    return D
end

function kernel_2mm_vectorized(
    alpha::T,
    beta::T,
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    C::AbstractMatrix{T},
    D::AbstractMatrix{T},
) where {T}
    return (alpha .* A * B * C) .+ (beta .* D)
end

function run_2mm_benchmark!(results, backend)
    N = 2048

    α = 0.01f0
    β = 0.5f0
    A = rand(Float32, N, N)
    B = rand(Float32, N, N)
    C = rand(Float32, N, N)
    D = rand(Float32, N, N)

    run_benchmark!(
        results,
        "2mm [$(N)]",
        backend,
        kernel_2mm,
        (α, β, A, B, C, D),
        kernel_2mm_vectorized;
        track_numbers=true,
    )
    return nothing
end

function kernel_3mm(
    A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T}
) where {T}
    E = similar(A, T, size(A, 1), size(B, 2))
    F = similar(A, T, size(C, 1), size(D, 2))
    G = similar(A, T, size(E, 1), size(F, 2))
    @trace for i in axes(A, 1)
        @trace for j in axes(B, 2)
            E[i, j] = 0
            @trace for k in axes(A, 2)
                E[i, j] += A[i, k] * B[k, j]
            end
        end
    end
    @trace for i in axes(C, 1)
        @trace for j in axes(D, 2)
            F[i, j] = 0
            @trace for k in axes(C, 2)
                F[i, j] += C[i, k] * D[k, j]
            end
        end
    end
    @trace for i in axes(E, 1)
        @trace for j in axes(F, 2)
            G[i, j] = 0
            @trace for k in axes(E, 2)
                G[i, j] += E[i, k] * F[k, j]
            end
        end
    end
    return G
end

function kernel_3mm_vectorized(
    A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T}
) where {T}
    return (A * B) * (C * D)
end

function run_3mm_benchmark!(results, backend)
    P, Q, R, S = 256, 1024, 2048, 4096

    A = rand(Float32, P, Q)
    B = rand(Float32, Q, R)
    C = rand(Float32, R, S)
    D = rand(Float32, S, P)

    run_benchmark!(
        results,
        "3mm [$(P), $(Q), $(R), $(S)]",
        backend,
        kernel_3mm,
        (A, B, C, D),
        kernel_3mm_vectorized;
        track_numbers=true,
    )
    return nothing
end

function kernel_atax(A::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    tmp = similar(A, T, size(A, 1))
    y = similar(A, T, size(A, 2))
    @trace for i in axes(A, 2)
        y[i] = 0
    end
    @trace for i in axes(A, 1)
        tmp[i] = 0
        @trace for j in axes(A, 2)
            tmp[i] += A[i, j] * x[j]
        end
        @trace for j in axes(A, 2)
            y[j] += A[i, j] * tmp[i]
        end
    end
    return y
end

function run_atax_benchmark!(results, backend)
    N = 2048

    A = rand(Float32, N, N)
    x = rand(Float32, N)

    run_benchmark!(
        results,
        "atax [$(N)]",
        backend,
        kernel_atax,
        (A, x),
        (A, x) -> A' * (A * x);
        track_numbers=true,
    )
    return nothing
end

function kernel_bicg(
    A::AbstractMatrix{T}, p::AbstractVector{T}, r::AbstractVector{T}
) where {T}
    s = similar(A, T, size(A, 2))
    q = similar(A, T, size(A, 1))

    @trace for i in axes(A, 2)
        s[i] = 0
    end

    @trace for i in axes(A, 1)
        q[i] = 0
        @trace for j in axes(A, 2)
            s[j] += r[i] * A[i, j]
            q[i] += A[i, j] * p[j]
        end
    end

    return s, q
end

function run_bicg_benchmark!(results, backend)
    N, M = 2048, 4096

    A = rand(Float32, N, M)
    p = rand(Float32, M)
    r = rand(Float32, N)

    run_benchmark!(
        results,
        "bicg [$(N), $(M)]",
        backend,
        kernel_bicg,
        (A, p, r),
        (A, p, r) -> (A' * r, A * p);
        track_numbers=true,
    )
    return nothing
end

function kernel_doitgen(A::AbstractArray{T,3}, x::AbstractArray{T,2}) where {T}
    R, Q, P = size(A)
    sum = similar(A, T, P)
    @trace for r in 1:R
        @trace for q in 1:Q
            @trace for p in 1:P
                sum[p] = 0
                @trace for s in 1:P
                    sum[p] += A[r, q, s] * x[s, p]
                end
            end
            @trace for p in 1:P
                A[r, q, p] = sum[p]
            end
        end
    end
    return A
end

function kernel_doitgen_vectorized(A::AbstractArray{T,3}, x::AbstractArray{T,2}) where {T}
    R, Q, S = size(A)
    return reshape(reshape(A, R * Q, S) * x, R, Q, S)
end

function run_doitgen_benchmark!(results, backend)
    R, Q, P = 256, 1024, 512

    A = rand(Float32, R, Q, P)
    x = rand(Float32, P, P)

    run_benchmark!(
        results,
        "doitgen [$(R), $(Q), $(P)]",
        backend,
        kernel_doitgen,
        (A, x),
        kernel_doitgen_vectorized;
        track_numbers=true,
    )
    return nothing
end

function kernel_mvt(
    x1::AbstractVector{T},
    x2::AbstractVector{T},
    y1::AbstractVector{T},
    y2::AbstractVector{T},
    A::AbstractMatrix{T},
) where {T}
    @trace for i in axes(x1, 1)
        @trace for j in axes(y1, 1)
            x1[i] += A[i, j] * y1[j]
        end
    end

    @trace for i in axes(x2, 1)
        @trace for j in axes(y2, 1)
            x2[i] += A[j, i] * y2[j]
        end
    end

    return x1, x2
end

function run_mvt_benchmark!(results, backend)
    N = 4096

    x1 = rand(Float32, N)
    x2 = rand(Float32, N)
    y1 = rand(Float32, N)
    y2 = rand(Float32, N)
    A = rand(Float32, N, N)

    run_benchmark!(
        results,
        "mvt [$(N)]",
        backend,
        kernel_mvt,
        (x1, x2, y1, y2, A),
        (x1, x2, y1, y2, A) -> (x1 .+ A * y1, x2 .+ A' * y2);
        track_numbers=true,
    )
    return nothing
end

function run_linalg_kernel_benchmarks!(results, backend)
    run_2mm_benchmark!(results, backend)
    run_3mm_benchmark!(results, backend)
    run_atax_benchmark!(results, backend)
    run_bicg_benchmark!(results, backend)
    run_doitgen_benchmark!(results, backend)
    run_mvt_benchmark!(results, backend)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    backend = get_backend()
    results = Dict()
    run_linalg_kernel_benchmarks!(results, backend)
end
