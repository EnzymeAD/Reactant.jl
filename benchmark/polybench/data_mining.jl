using Reactant
Reactant.allowscalar(true)

include("common.jl")

function correlation(D::AbstractMatrix{T}) where {T}
    n, m = size(D)
    mean = zeros(T, m)
    stddev = zeros(T, m)
    corr = zeros(T, m, m)

    @trace for j in 1:m
        mean[j] = false
        @trace for i in 1:n
            mean[j] += D[i, j]
        end
        mean[j] /= n
    end

    @trace for j in 1:m
        stddev[j] = false
        @trace for i in 1:n
            stddev[j] += (D[i, j] - mean[j]) * (D[i, j] - mean[j])
        end
        stddev[j] /= n
        stddev[j] = sqrt(stddev[j])
        stddev[j] = ifelse(stddev[j] <= T(0.1), T(1.0), stddev[j])
    end

    @trace for i in 1:n
        @trace for j in 1:m
            D[i, j] -= mean[j]
            D[i, j] /= T(sqrt(n)) * stddev[j]
        end
    end

    @trace for i in 1:m
        corr[i, i] = true
        # NOTE: small change here to avoid dynamic bounds (i + 1):m
        @trace for j in 1:m
            corr[i, j] = false
            @trace for k in 1:n
                corr[i, j] += D[k, i] * D[k, j]
            end
        end
    end

    return corr
end

function correlation_vectorized(D::AbstractMatrix{T}) where {T}
    n, m = size(D)
    μ = sum(D; dims=1) ./ n
    X = D .- μ
    # compute variance via dot product (X'X diagonal)
    σ = sqrt.(sum(X .^ 2; dims=1))   # 1 × cols
    σ = ifelse.(σ .<= T(0.1), T(1.0), σ)
    Xn = X ./ (σ ./ sqrt(T(n - 1)))
    return (Xn' * Xn) ./ (n - 1)
end

function covariance(D::AbstractMatrix{T}) where {T}
    n, m = size(D)
    mean = zeros(T, m)
    cov = zeros(T, m, m)

    @trace for j in 1:m
        mean[j] = false
        @trace for i in 1:n
            mean[j] += D[i, j]
        end
        mean[j] /= n
    end

    @trace for i in 1:n
        @trace for j in 1:m
            D[i, j] -= mean[j]
        end
    end

    @trace for i in 1:m
        # NOTE: small change here to avoid dynamic bounds i:m
        @trace for j in 1:m
            cov[i, j] = false
            @trace for k in 1:n
                cov[i, j] += D[k, i] * D[k, j]
            end
            cov[i, j] /= n - true
        end
    end

    return cov
end

function covariance_vectorized(D::AbstractMatrix{T}) where {T}
    n, m = size(D)
    mean = sum(D; dims=1) ./ n
    X = D .- mean
    # cov[i,j] = sum(X[:,i] .* X[:,j]) / (n-1)
    return (X' * X) ./ (n - 1)       # m×m
end

function run_data_mining_benchmarks!(results, backend)
    N = 2048
    A = rand(Float32, N, N)

    run_benchmark!(
        results,
        "correlation [$(N), $(N)]",
        backend,
        correlation,
        (A,),
        correlation_vectorized,
    )
    run_benchmark!(
        results, "covariance [$(N), $(N)]", backend, covariance, (A,), covariance_vectorized
    )

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    backend = get_backend()
    results = Dict()
    run_data_mining_benchmarks!(results, backend)
end
