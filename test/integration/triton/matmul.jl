using PythonCall, Reactant, Test

pyimport("sys").path.append(@__DIR__)

matmul_kernel = pyimport("matmul").matmul_kernel

const RunningOnCUDA = contains(string(Reactant.devices()[1]), "CUDA")

function matmul_triton(a::AbstractMatrix{T}, b::AbstractMatrix{T}) where {T}
    # a: [M, K] --> aᵀ: [K, M]
    # b: [K, N] --> bᵀ: [N, K]
    # c: a × b [M, N] --> cᵀ: bᵀ × aᵀ [N, M]
    a_transposed = permutedims(a, (2, 1)) # match python array layout
    b_transposed = permutedims(b, (2, 1)) # match python array layout
    @assert size(b_transposed, 2) == size(a_transposed, 1) "Inner dimensions must match \
                                                            for matmul"
    M, K = size(b_transposed)
    K, N = size(a_transposed)

    out = similar(a_transposed, T, M, N) # cᵀ

    matmul_kernel(
        b_transposed,
        a_transposed,
        out,
        M,
        N,
        K,
        Reactant.rowmajor_stride(b_transposed, 1),
        Reactant.rowmajor_stride(b_transposed, 2),
        Reactant.rowmajor_stride(a_transposed, 1),
        Reactant.rowmajor_stride(a_transposed, 2),
        Reactant.rowmajor_stride(out, 1),
        Reactant.rowmajor_stride(out, 2),
        64,
        256,
        32,
        8;
        grid=(cld(M, 64) * cld(N, 256),),
        num_stages=4,
        num_warps=4,
    )

    return permutedims(out, (2, 1))
end

@testset "matmul" begin
    if RunningOnCUDA
        @testset for M in (4, 32, 256, 1024),
            K in (4, 32, 512, 2048),
            N in (4, 32, 256, 1024)

            a = Reactant.to_rarray(rand(Float32, M, K))
            b = Reactant.to_rarray(rand(Float32, K, N))

            # XXX: shared_memory????
            # XXX: seems to work correctly for small matrices
            @test_broken @jit(matmul_triton(a, b)) ≈ @jit(a * b)
        end
    end
end
