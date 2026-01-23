using Reactant
Reactant.allowscalar(true)

include("common.jl")

# TODO: adi --> needs prefix sum raising to work
# TODO: seidel_2d --> the loops are sequential in nature. (prefix sum raising might help?)

function fdtd_2d(
    EX::AbstractMatrix{T},
    EY::AbstractMatrix{T},
    HZ::AbstractMatrix{T},
    fict::AbstractVector{T},
) where {T}
    NX, NY = size(EX)

    @trace for t in axes(fict, 1)
        @trace for j in axes(EY, 2)
            EY[1, j] = fict[t]
        end
        @trace for i in 2:NX
            @trace for j in 1:NY
                EY[i, j] = EY[i, j] - T(0.5) * (HZ[i, j] - HZ[i - 1, j])
            end
        end
        @trace for i in 1:NX
            @trace for j in 2:NY
                EX[i, j] = EX[i, j] - T(0.5) * (HZ[i, j] - HZ[i, j - 1])
            end
        end
        @trace for i in 1:(NX - 1)
            @trace for j in 1:(NY - 1)
                HZ[i, j] =
                    HZ[i, j] - T(0.7) * (EX[i, j + 1] - EX[i, j] + EY[i + 1, j] - EY[i, j])
            end
        end
    end

    return HZ
end

function fdtd_2d_vectorized(
    EX::AbstractMatrix{T},
    EY::AbstractMatrix{T},
    HZ::AbstractMatrix{T},
    fict::AbstractVector{T},
) where {T}
    NX, NY = size(EX)

    @trace track_numbers = false for t in axes(fict, 1)
        # Set boundary condition for EY first row
        fill!(view(EY, 1, :), fict[t])

        # Update EY using array slicing (vectorized operation)
        EY[2:NX, 1:NY] =
            EY[2:NX, 1:NY] .- T(0.5) .* (HZ[2:NX, 1:NY] .- HZ[1:(NX - 1), 1:NY])

        # Update EX using array slicing (vectorized operation)
        EX[1:NX, 2:NY] =
            EX[1:NX, 2:NY] .- T(0.5) .* (HZ[1:NX, 2:NY] .- HZ[1:NX, 1:(NY - 1)])

        # Update HZ using array slicing (vectorized operation)
        HZ[1:(NX - 1), 1:(NY - 1)] =
            HZ[1:(NX - 1), 1:(NY - 1)] .-
            T(0.7) .* (
                EX[1:(NX - 1), 2:NY] .- EX[1:(NX - 1), 1:(NY - 1)] .+
                EY[2:NX, 1:(NY - 1)] .- EY[1:(NX - 1), 1:(NY - 1)]
            )
    end

    return HZ
end

function run_fdtd_2d_benchmark!(results, backend)
    NX, NY, TMAX = 1024, 2048, 256

    EX = rand(Float32, NX, NY)
    EY = rand(Float32, NX, NY)
    HZ = rand(Float32, NX, NY)
    fict = rand(Float32, TMAX)

    run_benchmark!(
        results,
        "fdtd_2d [$(NX), $(NY), $(TMAX)]",
        backend,
        fdtd_2d,
        (EX, EY, HZ, fict),
        fdtd_2d_vectorized,
    )
    return nothing
end

function heat_3d(tsteps::Int, A::AbstractArray{T,3}, B::AbstractArray{T,3}) where {T}
    N = size(A, 1)

    @trace for _ in 1:tsteps
        @trace for i in 2:(N - 1)
            @trace for j in 2:(N - 1)
                @trace for k in 2:(N - 1)
                    B[i, j, k] = (
                        T(0.125) * (A[i + 1, j, k] - T(2.0) * A[i, j, k] + A[i - 1, j, k]) +
                        T(0.125) * (A[i, j + 1, k] - T(2.0) * A[i, j, k] + A[i, j - 1, k]) +
                        T(0.125) * (A[i, j, k + 1] - T(2.0) * A[i, j, k] + A[i, j, k - 1]) +
                        A[i, j, k]
                    )
                end
            end
        end
        @trace for i in 2:(N - 1)
            @trace for j in 2:(N - 1)
                @trace for k in 2:(N - 1)
                    A[i, j, k] = (
                        T(0.125) * (B[i + 1, j, k] - T(2.0) * B[i, j, k] + B[i - 1, j, k]) +
                        T(0.125) * (B[i, j + 1, k] - T(2.0) * B[i, j, k] + B[i, j - 1, k]) +
                        T(0.125) * (B[i, j, k + 1] - T(2.0) * B[i, j, k] + B[i, j, k - 1]) +
                        B[i, j, k]
                    )
                end
            end
        end
    end
    return A, B
end

function heat_3d_vectorized(
    tsteps::Int, A::AbstractArray{T,3}, B::AbstractArray{T,3}
) where {T}
    N = size(A, 1)
    @trace track_numbers = false for _ in 1:tsteps
        # Update B using array slicing (vectorized operation)
        B[2:(N - 1), 2:(N - 1), 2:(N - 1)] = (
            T(0.125) .* (
                A[3:N, 2:(N - 1), 2:(N - 1)] .-
                T(2.0) .* A[2:(N - 1), 2:(N - 1), 2:(N - 1)] .+
                A[1:(N - 2), 2:(N - 1), 2:(N - 1)]
            ) .+
            T(0.125) .* (
                A[2:(N - 1), 3:N, 2:(N - 1)] .-
                T(2.0) .* A[2:(N - 1), 2:(N - 1), 2:(N - 1)] .+
                A[2:(N - 1), 1:(N - 2), 2:(N - 1)]
            ) .+
            T(0.125) .* (
                A[2:(N - 1), 2:(N - 1), 3:N] .-
                T(2.0) .* A[2:(N - 1), 2:(N - 1), 2:(N - 1)] .+
                A[2:(N - 1), 2:(N - 1), 1:(N - 2)]
            ) .+ A[2:(N - 1), 2:(N - 1), 2:(N - 1)]
        )

        # Update A using array slicing (vectorized operation)
        A[2:(N - 1), 2:(N - 1), 2:(N - 1)] = (
            T(0.125) .* (
                B[3:N, 2:(N - 1), 2:(N - 1)] .-
                T(2.0) .* B[2:(N - 1), 2:(N - 1), 2:(N - 1)] .+
                B[1:(N - 2), 2:(N - 1), 2:(N - 1)]
            ) .+
            T(0.125) .* (
                B[2:(N - 1), 3:N, 2:(N - 1)] .-
                T(2.0) .* B[2:(N - 1), 2:(N - 1), 2:(N - 1)] .+
                B[2:(N - 1), 1:(N - 2), 2:(N - 1)]
            ) .+
            T(0.125) .* (
                B[2:(N - 1), 2:(N - 1), 3:N] .-
                T(2.0) .* B[2:(N - 1), 2:(N - 1), 2:(N - 1)] .+
                B[2:(N - 1), 2:(N - 1), 1:(N - 2)]
            ) .+ B[2:(N - 1), 2:(N - 1), 2:(N - 1)]
        )
    end
    return A, B
end

function run_heat_3d_benchmark!(results, backend)
    tsteps, N = 256, 128

    A = rand(Float32, N, N, N)
    B = rand(Float32, N, N, N)

    run_benchmark!(
        results,
        "heat_3d [$(N), $(N), $(N), $(tsteps)]",
        backend,
        heat_3d,
        (tsteps, A, B),
        heat_3d_vectorized,
    )
    return nothing
end

function jacobi_1d(tsteps::Int, A::AbstractVector{T}, B::AbstractVector{T}) where {T}
    N = size(A, 1)
    @trace for _ in 1:tsteps
        @trace for i in 2:(N - 1)
            B[i] = T(1 / 3) * (A[i - 1] + A[i] + A[i + 1])
        end
        @trace for i in 2:(N - 1)
            A[i] = T(1 / 3) * (B[i - 1] + B[i] + B[i + 1])
        end
    end
    return A, B
end

function jacobi_1d_vectorized(
    tsteps::Int, A::AbstractVector{T}, B::AbstractVector{T}
) where {T}
    N = size(A, 1)
    @trace track_numbers = false for _ in 1:tsteps
        # Update B using array slicing (vectorized operation)
        B[2:(N - 1)] = T(1 / 3) .* (
            A[1:(N - 2)] .+   # left neighbor
            A[2:(N - 1)] .+   # center
            A[3:N]          # right neighbor
        )

        # Update A using array slicing (vectorized operation)
        A[2:(N - 1)] = T(1 / 3) .* (
            B[1:(N - 2)] .+   # left neighbor
            B[2:(N - 1)] .+   # center
            B[3:N]          # right neighbor
        )
    end
    return A, B
end

function run_jacobi_1d_benchmark!(results, backend)
    tsteps, N = 1024, 2048

    A = rand(Float32, N)
    B = rand(Float32, N)

    run_benchmark!(
        results,
        "jacobi_1d [$(N), $(tsteps)]",
        backend,
        jacobi_1d,
        (tsteps, A, B),
        jacobi_1d_vectorized,
    )
    return nothing
end

function jacobi_2d(tsteps::Int, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    N = size(A, 1)
    @trace for t in 1:tsteps
        @trace for i in 2:(N - 1)
            @trace for j in 2:(N - 1)
                B[i, j] = (
                    T(0.2) *
                    (A[i, j] + A[i, j - 1] + A[i, 1 + j] + A[1 + i, j] + A[i - 1, j])
                )
            end
        end
        @trace for i in 2:(N - 1)
            @trace for j in 2:(N - 1)
                A[i, j] = (
                    T(0.2) *
                    (B[i, j] + B[i, j - 1] + B[i, 1 + j] + B[1 + i, j] + B[i - 1, j])
                )
            end
        end
    end
    return A, B
end

function jacobi_2d_vectorized(
    tsteps::Int, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where {T}
    N = size(A, 1)
    @trace track_numbers = false for t in 1:tsteps
        # Update B using array slicing (vectorized operation)
        B[2:(N - 1), 2:(N - 1)] =
            T(0.2) .* (
                A[2:(N - 1), 2:(N - 1)] .+   # center
                A[2:(N - 1), 1:(N - 2)] .+   # left
                A[2:(N - 1), 3:N] .+       # right
                A[3:N, 2:(N - 1)] .+       # bottom
                A[1:(N - 2), 2:(N - 1)]      # top
            )

        # Update A using array slicing (vectorized operation)
        A[2:(N - 1), 2:(N - 1)] =
            T(0.2) .* (
                B[2:(N - 1), 2:(N - 1)] .+   # center
                B[2:(N - 1), 1:(N - 2)] .+   # left
                B[2:(N - 1), 3:N] .+       # right
                B[3:N, 2:(N - 1)] .+       # bottom
                B[1:(N - 2), 2:(N - 1)]      # top
            )
    end
    return A, B
end

function run_jacobi_2d_benchmark!(results, backend)
    tsteps, N = 1024, 512

    A = rand(Float32, N, N)
    B = rand(Float32, N, N)

    run_benchmark!(
        results,
        "jacobi_2d [$(N), $(N), $(tsteps)]",
        backend,
        jacobi_2d,
        (tsteps, A, B),
        jacobi_2d_vectorized,
    )
    return nothing
end

function run_stencil_benchmarks!(results, backend)
    run_heat_3d_benchmark!(results, backend)
    run_jacobi_1d_benchmark!(results, backend)
    run_jacobi_2d_benchmark!(results, backend)
    run_fdtd_2d_benchmark!(results, backend)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    backend = get_backend()
    results = Dict()
    run_stencil_benchmarks!(results, backend)
end
